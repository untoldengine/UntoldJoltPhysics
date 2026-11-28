//
//  JoltSoftBody.swift
//  UntoldJoltPhysics
//
//  Soft bodies — particles joined by distance constraints, solved by Jolt's
//  own XPBD, colliding both ways with the rigid bodies — through the
//  plugin's side channel. The engine's physics seam has no soft-body
//  vocabulary, so a game builds one here from its own particle graph and
//  reads the vertex positions back every frame to drive a mesh. A body can
//  also be skinned to a skeleton: the game feeds the joints before every
//  step and Jolt keeps each particle near where the skin puts it.
//

import CJoltBridge
import simd
import UntoldEngine

/// One joint's pull on a skinned vertex.
public struct JoltSoftBodySkinInfluence: Sendable {
    /// Index into `JoltSoftBodySkin.bindJointWorldMatrices`.
    public var joint: UInt32
    public var weight: Float

    public init(joint: UInt32, weight: Float) {
        self.joint = joint
        self.weight = weight
    }
}

/// A particle tied to the skeleton (Jolt's skinned constraint): every
/// frame the skin moves its rest position with the joints, and the solver
/// keeps the simulated particle within `maxDistance` of that point and
/// outside a back-stop sphere behind it.
public struct JoltSoftBodySkinnedVertex: Sendable {
    public var vertex: UInt32
    /// One to four joints; the bridge normalises the weights.
    public var influences: [JoltSoftBodySkinInfluence]
    /// How far the particle may stray from its skinned position. 0 hard
    /// skins it: it sits exactly there and its inverse mass is forced to
    /// 0, so colliders never fight the skin over it.
    public var maxDistance: Float
    /// The back-stop sphere starts this far behind the skinned position,
    /// against the skin's normal, and keeps the particle outside the body
    /// the cloth is worn over. Disabled unless it is below `maxDistance`
    /// (times the multiplier `setSoftBodySkin` sets). The normal comes from
    /// the faces around the vertex whose three vertices are all skinned, so
    /// a back-stopped vertex needs one or `addSoftBody` refuses the body.
    public var backstopDistance: Float = .greatestFiniteMagnitude
    /// Radius of that sphere; 40 m approximates a plane.
    public var backstopRadius: Float = 40

    public init(vertex: UInt32, influences: [JoltSoftBodySkinInfluence], maxDistance: Float) {
        self.vertex = vertex
        self.influences = influences
        self.maxDistance = maxDistance
    }
}

/// How a soft body hangs on a skeleton.
public struct JoltSoftBodySkin: Sendable {
    /// The joints' world transforms in the pose the vertices were authored
    /// in (the bind pose); a vertex's influences index this array, and
    /// every feed supplies the same joints in the same order.
    public var bindJointWorldMatrices: [simd_float4x4]
    public var vertices: [JoltSoftBodySkinnedVertex]

    public init(bindJointWorldMatrices: [simd_float4x4], vertices: [JoltSoftBodySkinnedVertex]) {
        self.bindJointWorldMatrices = bindJointWorldMatrices
        self.vertices = vertices
    }
}

public struct JoltSoftBodyDescriptor: Sendable {
    /// Particle positions, relative to `position`.
    public var vertices: [SIMD3<Float>]
    /// Per particle; 0 pins the particle in place.
    public var inverseMasses: [Float]
    /// Distance constraints between particle indices.
    public var edges: [SIMD2<UInt32>]
    /// Inverse stiffness of the edges, in m/N; 0 is rigid.
    public var compliance: Float = 1e-6
    /// Per-edge compliance overriding `compliance` when set.
    public var edgeCompliances: [Float]? = nil
    /// Optional triangles: with them the body has a surface, so a sphere
    /// cannot slip between particles (and continuous collision applies).
    /// For a skinned body, wind them counter-clockwise seen from outside
    /// the cloth: their normal is the skin's, and the back-stop lies behind
    /// it, inside the body the cloth is worn over.
    public var faces: [SIMD3<UInt32>] = []
    public var position: SIMD3<Float> = .zero
    public var layer: UInt32 = 0
    /// Solver iterations per step (Jolt substeps the body by this count).
    public var iterations: UInt32 = 10
    public var linearDamping: Float = 1.0
    /// Collision radius of a particle. Rigid bodies collide with the
    /// particles as spheres, so this radius plus the other body's must
    /// cover half the particle spacing or that body slips through the gaps;
    /// faces alone do not close them.
    public var vertexRadius: Float = 0.004
    public var friction: Float = 0.4
    public var restitution: Float = 0.0
    public var gravityFactor: Float = 1.0
    /// Cloth from triangles: the stretch, shear and dihedral bend
    /// constraints are derived from `faces` (Jolt's CreateConstraints) with
    /// `compliance`, `shearCompliance` and `bendCompliance`; `edges` is
    /// then ignored.
    public var constraintsFromFaces = false
    public var shearCompliance: Float = 1e-5
    /// Inverse bending stiffness; larger folds more easily.
    public var bendCompliance: Float = 1e-3
    /// Ceiling on a vertex's speed (m/s; 0 = Jolt's default): bounds the
    /// energy a resolved overlap or a yanked pin can put into the body.
    public var maxLinearVelocity: Float = 0
    /// How bending resists when the constraints come from the faces.
    public enum BendType: Int32, Sendable {
        /// The angle between neighbouring triangles: stiffer, and less
        /// stable when pinned vertices move fast.
        case dihedral = 0
        /// A distance across the shared edge: robust.
        case distance = 1
    }

    public var bendType: BendType = .dihedral
    /// Skinning to a skeleton; needs `faces` (the back-stop normals come
    /// from them). The game then feeds the joints before every step with
    /// `JoltPhysicsBackend.skinSoftBody`; the body is created skinned to
    /// its bind pose, so a step before the first feed is sound.
    public var skin: JoltSoftBodySkin? = nil
    /// Whether Jolt may put the body to sleep at rest (its default). Jolt
    /// never wakes a body for its skin, so cloth on a live character
    /// passes false rather than rely on every feed waking it.
    public var allowSleeping = true
    /// Ragdoll parts push this body and it never pushes them: a swinging
    /// dynamic arm drags the cloth along by friction while receiving nothing
    /// back, a kinematic (animated) part only pushes. Every other body
    /// collides both ways. The one gap is Jolt's continuous collision: a
    /// dynamic part moving more than three quarters of its inner radius in
    /// a sub-step is still held at the cloth for that sub-step (no impulse,
    /// but the rest of its travel that sub-step is lost); raise
    /// `collisionSteps` or turn `continuousCollision` off if it shows.
    public var oneWayAgainstRagdolls = false

    public init(vertices: [SIMD3<Float>], inverseMasses: [Float], edges: [SIMD2<UInt32>]) {
        self.vertices = vertices
        self.inverseMasses = inverseMasses
        self.edges = edges
    }

    /// A cloth: constraints from the triangles.
    public init(vertices: [SIMD3<Float>], inverseMasses: [Float], faces: [SIMD3<UInt32>], compliance: Float, shearCompliance: Float, bendCompliance: Float) {
        self.vertices = vertices
        self.inverseMasses = inverseMasses
        edges = []
        self.faces = faces
        self.compliance = compliance
        self.shearCompliance = shearCompliance
        self.bendCompliance = bendCompliance
        constraintsFromFaces = true
    }
}

/// A soft body in the world; remove it with `JoltPhysicsBackend.removeSoftBody`.
/// Frame thread only: the read-back scratch it owns is shared by every
/// call made on it.
public final class JoltSoftBody: @unchecked Sendable {
    let id: ujolt_body_id
    public let vertexCount: Int
    /// The joints the skin was built with (0 for an unskinned body); every
    /// feed must supply exactly this many.
    public let jointCount: Int
    public var isSkinned: Bool { jointCount > 0 }
    /// xyz triples for the bridge, so a read-back allocates nothing.
    var positionScratch: [Float]
    var normalScratch: [Float]

    init(id: ujolt_body_id, vertexCount: Int, jointCount: Int) {
        self.id = id
        self.vertexCount = vertexCount
        self.jointCount = jointCount
        positionScratch = Array(repeating: 0, count: vertexCount * 3)
        normalScratch = Array(repeating: 0, count: vertexCount * 3)
    }
}

/// A rigid translation, the only transform the bridge frame needs.
private func translation(_ offset: SIMD3<Float>) -> simd_float4x4 {
    var m = matrix_identity_float4x4
    m.columns.3 = SIMD4<Float>(offset, 1)
    return m
}

extension JoltPhysicsBackend {
    /// Adds a soft body; nil if the descriptor is malformed (an edge or face
    /// index out of range, a zero-length edge, a skin without faces or
    /// joints, a skinned vertex pulled by no joint, by more than four or by
    /// a joint the skin does not have, a back-stopped vertex with no face
    /// whose three vertices are all skinned). Frame thread.
    public func addSoftBody(_ descriptor: JoltSoftBodyDescriptor) -> JoltSoftBody? {
        guard !descriptor.vertices.isEmpty, descriptor.inverseMasses.count == descriptor.vertices.count else { return nil }
        var positions: [Float] = []
        positions.reserveCapacity(descriptor.vertices.count * 3)
        for v in descriptor.vertices { positions += [v.x, v.y, v.z] }
        var edges: [UInt32] = []
        edges.reserveCapacity(descriptor.edges.count * 2)
        for e in descriptor.edges { edges += [e.x, e.y] }
        var faces: [UInt32] = []
        faces.reserveCapacity(descriptor.faces.count * 3)
        for f in descriptor.faces { faces += [f.x, f.y, f.z] }
        let compliances = descriptor.edgeCompliances ?? []

        var invBind: [simd_float4x4] = []
        var skinned: [ujolt_soft_body_skin_vertex] = []
        if let skin = descriptor.skin {
            // The back-stop normals come from the faces; Jolt takes at most
            // four joints per vertex, and one pulled by none would be
            // skinned to the origin.
            guard !descriptor.faces.isEmpty, !skin.bindJointWorldMatrices.isEmpty,
                  skin.vertices.allSatisfy({ (1 ... 4).contains($0.influences.count) }) else { return nil }
            // The vertices are handed over relative to `position`, so the
            // bind matrices move into that frame before they are inverted.
            let toBodyFrame = translation(-descriptor.position)
            invBind = skin.bindJointWorldMatrices.map { (toBodyFrame * $0).inverse }
            skinned.reserveCapacity(skin.vertices.count)
            for vertex in skin.vertices {
                var c = ujolt_soft_body_skin_vertex()
                c.vertex = vertex.vertex
                var joints: [UInt32] = [0, 0, 0, 0]
                var weights: [Float] = [0, 0, 0, 0]
                for (k, influence) in vertex.influences.prefix(4).enumerated() {
                    joints[k] = influence.joint
                    weights[k] = influence.weight
                }
                c.joints = (joints[0], joints[1], joints[2], joints[3])
                c.weights = (weights[0], weights[1], weights[2], weights[3])
                c.max_distance = vertex.maxDistance
                c.backstop_distance = vertex.backstopDistance
                c.backstop_radius = vertex.backstopRadius
                skinned.append(c)
            }
        }

        var desc = ujolt_soft_body_desc()
        desc.vertex_count = UInt32(descriptor.vertices.count)
        desc.edge_count = UInt32(descriptor.edges.count)
        desc.face_count = UInt32(descriptor.faces.count)
        desc.compliance = descriptor.compliance
        desc.position = (descriptor.position.x, descriptor.position.y, descriptor.position.z)
        desc.layer = descriptor.layer
        desc.iterations = descriptor.iterations
        desc.linear_damping = descriptor.linearDamping
        desc.vertex_radius = descriptor.vertexRadius
        desc.friction = descriptor.friction
        desc.restitution = descriptor.restitution
        desc.gravity_factor = descriptor.gravityFactor
        desc.constraints_from_faces = descriptor.constraintsFromFaces ? 1 : 0
        desc.shear_compliance = descriptor.shearCompliance
        desc.bend_compliance = descriptor.bendCompliance
        desc.max_linear_velocity = descriptor.maxLinearVelocity
        desc.bend_type = descriptor.bendType.rawValue
        desc.joint_count = UInt32(invBind.count)
        desc.skinned_count = UInt32(skinned.count)
        desc.allow_sleeping = descriptor.allowSleeping ? 1 : 0
        desc.one_way_against_ragdolls = descriptor.oneWayAgainstRagdolls ? 1 : 0
        // Not an engine entity: its activations look like the environment's.
        desc.user_data = UInt64(Self.environmentEntity)

        let id: ujolt_body_id = positions.withUnsafeBufferPointer { p in
            descriptor.inverseMasses.withUnsafeBufferPointer { m in
                edges.withUnsafeBufferPointer { e in
                    faces.withUnsafeBufferPointer { f in
                        compliances.withUnsafeBufferPointer { c in
                            skinned.withUnsafeBufferPointer { s in
                                // simd_float4x4 is 16 contiguous floats,
                                // column-major: the layout the bridge reads.
                                invBind.withUnsafeBufferPointer { matrices in
                                    matrices.withMemoryRebound(to: Float.self) { b in
                                        desc.vertices = p.baseAddress
                                        desc.inv_masses = m.baseAddress
                                        desc.edges = e.baseAddress
                                        desc.faces = f.isEmpty ? nil : f.baseAddress
                                        desc.edge_compliances = c.isEmpty ? nil : c.baseAddress
                                        desc.skinned = s.isEmpty ? nil : s.baseAddress
                                        desc.inv_bind = b.isEmpty ? nil : b.baseAddress
                                        return ujolt_world_add_soft_body(worldHandle, &desc)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        guard id != UJOLT_INVALID_BODY else { return nil }
        return JoltSoftBody(id: id, vertexCount: Int(ujolt_world_soft_body_vertex_count(worldHandle, id)),
                            jointCount: skinned.isEmpty ? 0 : invBind.count)
    }

    /// Reads the world-space particle positions into `positions` (resized to
    /// the body's vertex count). Returns the count read. Frame thread,
    /// between steps.
    @discardableResult
    public func readSoftBodyVertices(_ body: JoltSoftBody, into positions: inout [SIMD3<Float>]) -> Int {
        if positions.count != body.vertexCount {
            positions = Array(repeating: .zero, count: body.vertexCount)
        }
        let read = body.positionScratch.withUnsafeMutableBufferPointer { buffer in
            Int(ujolt_world_read_soft_body_vertices(worldHandle, body.id, buffer.baseAddress, UInt32(body.vertexCount)))
        }
        for index in 0 ..< read {
            positions[index] = SIMD3<Float>(body.positionScratch[index * 3], body.positionScratch[index * 3 + 1], body.positionScratch[index * 3 + 2])
        }
        return read
    }

    /// Reads the world-space positions and the unit vertex normals
    /// (area-weighted over the body's faces; +Y for a vertex on none) into
    /// the two arrays, each resized to the vertex count. Nothing is
    /// allocated after the first call. Returns the count read. Frame
    /// thread, between steps.
    @discardableResult
    public func readSoftBodyVertices(_ body: JoltSoftBody, into positions: inout [SIMD3<Float>], normals: inout [SIMD3<Float>]) -> Int {
        if positions.count != body.vertexCount {
            positions = Array(repeating: .zero, count: body.vertexCount)
        }
        if normals.count != body.vertexCount {
            normals = Array(repeating: SIMD3<Float>(0, 1, 0), count: body.vertexCount)
        }
        let read = body.positionScratch.withUnsafeMutableBufferPointer { p in
            body.normalScratch.withUnsafeMutableBufferPointer { n in
                Int(ujolt_world_read_soft_body_vertices_and_normals(worldHandle, body.id, p.baseAddress, n.baseAddress, UInt32(body.vertexCount)))
            }
        }
        for index in 0 ..< read {
            positions[index] = SIMD3<Float>(body.positionScratch[index * 3], body.positionScratch[index * 3 + 1], body.positionScratch[index * 3 + 2])
            normals[index] = SIMD3<Float>(body.normalScratch[index * 3], body.normalScratch[index * 3 + 1], body.normalScratch[index * 3 + 2])
        }
        return read
    }

    /// Moves the given vertices to world positions and clears their
    /// velocity: how pinned vertices ride on something animated. Frame
    /// thread, between steps. Returns the count applied.
    @discardableResult
    public func setSoftBodyVertices(_ body: JoltSoftBody, indices: [UInt32], worldPositions: [SIMD3<Float>]) -> Int {
        let count = min(indices.count, worldPositions.count)
        guard count > 0 else { return 0 }
        // A non-finite pin would poison the body's bounds next step.
        guard worldPositions.prefix(count).allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else { return 0 }
        var flat = [Float](repeating: 0, count: count * 3)
        for i in 0 ..< count {
            flat[i * 3] = worldPositions[i].x
            flat[i * 3 + 1] = worldPositions[i].y
            flat[i * 3 + 2] = worldPositions[i].z
        }
        return indices.withUnsafeBufferPointer { indexPointer in
            flat.withUnsafeBufferPointer { positionPointer in
                Int(ujolt_world_set_soft_body_vertices(worldHandle, body.id, indexPointer.baseAddress, positionPointer.baseAddress, UInt32(count)))
            }
        }
    }

    /// Feeds this step's joints to a skinned body: one world transform per
    /// joint, in the skin's order. Jolt moves each skinned vertex's rest
    /// position with them and the step keeps the particles within their
    /// slack of it. `hardSkinAll` instead places every skinned vertex on
    /// its skinned position with no velocity — a teleport — and is the
    /// only hard reset there is; a body's first feed is one whatever is
    /// asked, since Jolt blends the skinned positions from the previous
    /// feed's and there is none yet. Frame thread, between
    /// steps: from `willStep`, never from inside a step. The matrices are
    /// handed over in place. False, and nothing applied, when the body is
    /// not skinned, the count is not the skin's joint count or a matrix is
    /// not finite.
    @discardableResult
    public func skinSoftBody(_ body: JoltSoftBody, jointWorldMatrices: [simd_float4x4], hardSkinAll: Bool = false) -> Bool {
        guard body.isSkinned, jointWorldMatrices.count == body.jointCount else { return false }
        return jointWorldMatrices.withUnsafeBufferPointer { matrices in
            matrices.withMemoryRebound(to: Float.self) { floats in
                ujolt_world_skin_soft_body(worldHandle, body.id, floats.baseAddress, UInt32(body.jointCount), hardSkinAll ? 1 : 0) != 0
            }
        }
    }

    /// Scales every skinned vertex's `maxDistance` (0 hard skins the whole
    /// body to its skin; 1 restores the descriptor's; below a vertex's
    /// `backstopDistance / maxDistance` its back-stop is off) and switches
    /// the skin constraints on or off (off, only the hard-skinned vertices
    /// still follow the skin and the rest hangs free). A non-finite
    /// multiplier applies nothing. Frame thread, between steps.
    public func setSoftBodySkin(_ body: JoltSoftBody, maxDistanceMultiplier: Float, enabled: Bool = true) {
        ujolt_world_set_soft_body_skin_params(worldHandle, body.id, maxDistanceMultiplier, enabled ? 1 : 0)
    }

    /// Whether Jolt is simulating the body (false once it sleeps or is
    /// removed).
    public func isSoftBodyActive(_ body: JoltSoftBody) -> Bool {
        ujolt_world_body_is_active(worldHandle, body.id) != 0
    }

    /// Puts the body to sleep or wakes it. Asleep, Jolt leaves it out of the
    /// step altogether — no solve, no collision of its own, no skinning: the
    /// soft-body jobs take the active soft bodies only — so a resting cloth
    /// stops costing, and its particles stay where they were. `skinSoftBody`
    /// and `setSoftBodySkin` wake it (a sleeping body would not follow what
    /// it was just given), and so does an awake rigid body reaching its
    /// bounds on a layer the two share: Jolt pairs an active rigid body with
    /// a sleeping soft body and wakes the soft body before it leaves the pair
    /// to the soft-body update, so a ball thrown at a sleeping shirt wakes
    /// it — and a ragdoll part sitting inside it would wake it every step. A
    /// body to be kept asleep goes on a layer nothing awake meets
    /// (`setSoftBodyLayer`). Frame thread, between steps.
    public func setSoftBodyActive(_ body: JoltSoftBody, _ active: Bool) {
        ujolt_world_set_body_active(worldHandle, body.id, active ? 1 : 0)
    }

    /// Moves the body to another engine collision layer (0 ... 31): the
    /// layer matrix then decides what meets it, from the next step. Frame
    /// thread, between steps.
    public func setSoftBodyLayer(_ body: JoltSoftBody, _ layer: UInt32) {
        ujolt_world_set_body_layer(worldHandle, body.id, layer)
    }

    public func removeSoftBody(_ body: JoltSoftBody) {
        ujolt_world_remove_body(worldHandle, body.id)
    }
}
