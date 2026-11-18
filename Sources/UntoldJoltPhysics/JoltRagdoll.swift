//
//  JoltRagdoll.swift
//  UntoldJoltPhysics
//
//  A ragdoll — Jolt's Ragdoll over swing-twist joints with motors — through
//  the plugin's side channel. The engine's physics seam has no constraint
//  vocabulary, so a game builds one here from its skeleton: one rigid part
//  per joint with the body origin at the joint pivot, each part either
//  kinematic (it follows the pose the game gives every frame) or dynamic
//  (simulated, its motor pulling it toward the pose the game drives). A
//  knockdown is every part dynamic with the motors off and some joint
//  friction, the mesh following the ragdoll; a hit reaction is the hit
//  subtree dynamic with its motors driving it toward the live animation
//  while the rest stays kinematic. Every part reports the ragdoll's entity
//  in contacts and rays; none is ever read back through the engine's
//  transform batch.
//

import CJoltBridge
import Foundation
import simd
import UntoldEngine

public struct JoltRagdollPart: Sendable {
    public var name: String
    /// The parent part's index, which must come earlier in the array; nil
    /// for the root, which must be part 0.
    public var parentIndex: Int?
    /// `.capsule(radius:height:)` with `height` the cylindrical section, as
    /// the engine's, `.sphere` or `.box` (a cylinder is taken too; a convex
    /// hull is refused).
    public var shape: PhysicsColliderShape
    /// The shape's centre in the part's frame. The body origin is the joint
    /// pivot, so a bone's capsule sits half a bone length along its axis.
    public var shapeOffset: SIMD3<Float> = .zero
    /// The shape's rotation in the part's frame (a capsule's axis is its
    /// local Y).
    public var shapeRotation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    /// Kilograms. Jolt then clamps every parent/child mass ratio to
    /// 0.8…1.2, keeping the total, so a chain does not blow up.
    public var mass: Float
    /// The neutral pose, world: the joint pivot.
    public var position: SIMD3<Float>
    /// The neutral pose, world: the joint frame.
    public var rotation: simd_quatf
    /// The joint to the parent, at this pivot, in the neutral pose: the
    /// twist axis, world, along this part's bone.
    public var twistAxis: SIMD3<Float>
    /// A perpendicular axis of that joint, made orthogonal to the twist
    /// axis if it is not.
    public var planeAxis: SIMD3<Float>
    /// The same two axes as the parent holds them in the neutral pose,
    /// world. Nil for the part's own: the limits are then centred on the
    /// neutral pose. Rotate them away from the part's axes to centre the
    /// cones elsewhere — a knee's on mid-flexion (turn them about the
    /// plane axis, the hinge pin), so one symmetric cone covers straight to
    /// fully bent and nothing beyond straight.
    public var parentTwistAxis: SIMD3<Float>?
    public var parentPlaneAxis: SIMD3<Float>?
    /// Twist about `twistAxis`, radians.
    public var twistRange: ClosedRange<Float> = -0.5 ... 0.5
    /// Swing limits, radians: how far the bone may tilt toward the normal
    /// (twist × plane) axis — a rotation about the plane axis, a hinge's
    /// bend when the plane axis is its pin.
    public var normalHalfConeAngle: Float = 0.5
    /// And how far toward the plane axis (a rotation about the normal).
    public var planeHalfConeAngle: Float = 0.5
    /// The motor spring, in Hz.
    public var motorFrequency: Float = 20
    /// The motor spring's damping ratio.
    public var motorDamping: Float = 2
    /// The most torque the motor may apply, N m.
    public var maxTorque: Float = 500
    /// N m of resistance at the joint while its motors are off.
    public var frictionTorque: Float = 0
    /// The part's surface friction against whatever it lands on (Jolt's
    /// bodies default to 0.2, which slides).
    public var friction: Float = 0.5

    public init(name: String, parentIndex: Int?, shape: PhysicsColliderShape, mass: Float,
                position: SIMD3<Float>, rotation: simd_quatf, twistAxis: SIMD3<Float>, planeAxis: SIMD3<Float>)
    {
        self.name = name
        self.parentIndex = parentIndex
        self.shape = shape
        self.mass = mass
        self.position = position
        self.rotation = rotation
        self.twistAxis = twistAxis
        self.planeAxis = planeAxis
    }
}

public struct JoltRagdollDescriptor: Sendable {
    public var parts: [JoltRagdollPart]
    /// Reported by contacts with, and ray hits on, every part.
    public var entity: EntityID = JoltPhysicsBackend.environmentEntity
    public var layer: UInt32 = 0
    public var gravityFactor: Float = 1
    public var linearDamping: Float = 0.05
    public var angularDamping: Float = 0.05
    /// Metres per second no part may exceed.
    public var maxLinearVelocity: Float = 50
    /// Whether the parts are in the world from creation; otherwise
    /// `JoltRagdoll.setActive(true)` puts them there.
    public var startActive = false
    /// Pairs of part indices that never collide with each other, on top of
    /// every parent/child pair and every pair overlapping in the neutral
    /// pose. A torso resting on its own thighs props a fallen body up:
    /// disable those pairs.
    public var disabledCollisionPairs: [(Int, Int)] = []
    /// Solver iterations for the joints, per step; 0 = the world's (Jolt's
    /// 10 velocity, 2 position). A long powered chain converges slowly:
    /// its outer links trail the animation until the joints get more.
    public var solverVelocitySteps = 0
    public var solverPositionSteps = 0

    public init(parts: [JoltRagdollPart]) {
        self.parts = parts
    }
}

public enum JoltMotorMode: Sendable {
    case off
    /// The joint's motors drive toward the last target `driveMotors` gave
    /// (the neutral pose before any).
    case position
}

/// A ragdoll in the world; remove it with `JoltPhysicsBackend.removeRagdoll`.
/// Poses are one world transform per part, rigid, with the body origin at
/// the joint pivot. Every call is frame-thread only, between world steps.
public final class JoltRagdoll: @unchecked Sendable {
    private(set) var handle: OpaquePointer?
    public let entity: EntityID
    public let partCount: Int
    public let partNames: [String]
    public let parentIndices: [Int?]
    /// One 4x4 per part, column-major, as the bridge wants them.
    private var poseScratch: [Float]

    init(handle: OpaquePointer, descriptor: JoltRagdollDescriptor) {
        self.handle = handle
        entity = descriptor.entity
        partCount = descriptor.parts.count
        partNames = descriptor.parts.map(\.name)
        parentIndices = descriptor.parts.map(\.parentIndex)
        poseScratch = Array(repeating: 0, count: descriptor.parts.count * 16)
    }

    /// False once the ragdoll has been removed; every call is then a no-op.
    public var isValid: Bool { handle != nil }

    /// Whether the parts are in the world.
    public var isActive: Bool {
        guard let handle else { return false }
        return ujolt_ragdoll_is_active(handle) != 0
    }

    /// Puts the parts and their joints into the world, or takes them out
    /// (they keep their state).
    public func setActive(_ active: Bool) {
        guard let handle else { return }
        ujolt_ragdoll_set_active(handle, active ? 1 : 0)
    }

    /// Teleports every part; the joints forget their last impulses and the
    /// ragdoll wakes when active. Ignored unless there is one transform per
    /// part.
    public func setPose(_ worldTransforms: [simd_float4x4], resetVelocities: Bool = true) {
        guard let handle else { return }
        withFlattened(worldTransforms) { ujolt_ragdoll_set_pose(handle, $0, resetVelocities ? 1 : 0) }
    }

    /// One velocity per part, world; `angular` nil sets none. Ignored unless
    /// the counts match.
    public func setVelocities(linear: [SIMD3<Float>], angular: [SIMD3<Float>]?) {
        guard let handle, linear.count == partCount, angular == nil || angular?.count == partCount else { return }
        var flatLinear: [Float] = []
        flatLinear.reserveCapacity(partCount * 3)
        for v in linear { flatLinear += [v.x, v.y, v.z] }
        var flatAngular: [Float] = []
        if let angular {
            flatAngular.reserveCapacity(partCount * 3)
            for v in angular { flatAngular += [v.x, v.y, v.z] }
        }
        flatLinear.withUnsafeBufferPointer { l in
            flatAngular.withUnsafeBufferPointer { a in
                ujolt_ragdoll_set_velocities(handle, l.baseAddress, flatAngular.isEmpty ? nil : a.baseAddress)
            }
        }
    }

    /// The pose the kinematic parts move to during the next step(s), with
    /// the velocity that implies; kept until replaced. Dynamic parts ignore
    /// it. Ignored unless there is one transform per part.
    public func setKinematicPose(_ worldTransforms: [simd_float4x4]) {
        guard let handle else { return }
        withFlattened(worldTransforms) { ujolt_ragdoll_set_kinematic_pose(handle, $0) }
    }

    /// Motor targets for every joint whose motors are in position mode: each
    /// part's rotation relative to its parent, taken from the pose. The next
    /// step consumes them. Ignored unless there is one transform per part.
    public func driveMotors(toward worldTransforms: [simd_float4x4]) {
        guard let handle else { return }
        withFlattened(worldTransforms) { ujolt_ragdoll_drive_motors(handle, $0) }
    }

    /// Makes a part (nil: every part) simulated, or kinematic when `dynamic`
    /// is false. A part made kinematic stops where it is until it is given
    /// a pose.
    public func setPartDynamic(_ index: Int?, _ dynamic: Bool) {
        guard let handle else { return }
        ujolt_ragdoll_set_part_dynamic(handle, index.map { Int32($0) } ?? -1, dynamic ? 1 : 0)
    }

    /// Adds m/s to the linear velocity of a dynamic part (nil = every
    /// dynamic part), on top of whatever it has: a shove of the whole body.
    public func addLinearVelocity(_ delta: SIMD3<Float>, toPart index: Int? = nil) {
        guard let handle else { return }
        var d: (Float, Float, Float) = (delta.x, delta.y, delta.z)
        withUnsafePointer(to: &d) { pointer in
            pointer.withMemoryRebound(to: Float.self, capacity: 3) {
                ujolt_ragdoll_add_linear_velocity(handle, index.map { Int32($0) } ?? -1, $0)
            }
        }
    }

    /// Gravity on a part (nil = every part): 1 is the world's, 0 none. A
    /// reacting limb the animation holds feels only the hit.
    public func setGravityFactor(_ index: Int?, _ factor: Float) {
        guard let handle else { return }
        ujolt_ragdoll_set_gravity_factor(handle, index.map { Int32($0) } ?? -1, factor)
    }

    /// Whether Jolt may put the parts to sleep when they come to rest (its
    /// default). A body settling through a slow topple can pause below the
    /// sleep threshold long enough to be frozen mid-fall: forbid sleeping
    /// until it lies.
    public func setAllowSleeping(_ allow: Bool) {
        guard let handle else { return }
        ujolt_ragdoll_set_allow_sleeping(handle, allow ? 1 : 0)
    }

    /// True while Jolt simulates the part; false once it sleeps or the
    /// parts are out of the world.
    public func isPartAwake(_ index: Int) -> Bool {
        guard let handle else { return false }
        return ujolt_ragdoll_part_is_awake(handle, Int32(index)) != 0
    }

    public func isPartDynamic(_ index: Int) -> Bool {
        guard let handle else { return false }
        return ujolt_ragdoll_part_is_dynamic(handle, Int32(index)) != 0
    }

    /// Switches the motors of a part's joint (nil: every joint; the root has
    /// none) and retunes them; a value of 0 keeps that setting.
    public func setMotors(_ index: Int?, mode: JoltMotorMode, frequency: Float = 0, damping: Float = 0,
                          maxTorque: Float = 0, frictionTorque: Float = 0)
    {
        guard let handle else { return }
        let cMode = mode == .position ? UJOLT_MOTOR_POSITION : UJOLT_MOTOR_OFF
        ujolt_ragdoll_set_motors(handle, index.map { Int32($0) } ?? -1, cMode, frequency, damping, maxTorque, frictionTorque)
    }

    /// Reads the current world transform of every part into `transforms`
    /// (resized to the part count). Returns the count read; 0 once removed.
    @discardableResult
    public func readPose(into transforms: inout [simd_float4x4]) -> Int {
        guard let handle else { return 0 }
        if transforms.count != partCount {
            transforms = Array(repeating: matrix_identity_float4x4, count: partCount)
        }
        let read = poseScratch.withUnsafeMutableBufferPointer { buffer in
            Int(ujolt_ragdoll_read_pose(handle, buffer.baseAddress, UInt32(partCount)))
        }
        for index in 0 ..< read {
            let base = index * 16
            transforms[index] = simd_float4x4(
                SIMD4<Float>(poseScratch[base], poseScratch[base + 1], poseScratch[base + 2], poseScratch[base + 3]),
                SIMD4<Float>(poseScratch[base + 4], poseScratch[base + 5], poseScratch[base + 6], poseScratch[base + 7]),
                SIMD4<Float>(poseScratch[base + 8], poseScratch[base + 9], poseScratch[base + 10], poseScratch[base + 11]),
                SIMD4<Float>(poseScratch[base + 12], poseScratch[base + 13], poseScratch[base + 14], poseScratch[base + 15])
            )
        }
        return read
    }

    /// An impulse in N s at a world point, on a dynamic part of an active
    /// ragdoll; nothing happens to a kinematic one.
    /// A part's joint rotation in its constraint space: identity where the
    /// parent's and the part's axes meet, twist about X, swing about Y (the
    /// normal axis) and Z (the plane axis). For tuning limits; nil for the
    /// root.
    public func jointRotation(ofPart index: Int) -> simd_quatf? {
        guard let handle else { return nil }
        var q: (Float, Float, Float, Float) = (0, 0, 0, 1)
        let ok = withUnsafeMutablePointer(to: &q) { pointer in
            pointer.withMemoryRebound(to: Float.self, capacity: 4) { ujolt_ragdoll_read_joint_rotation(handle, Int32(index), $0) }
        }
        return ok != 0 ? simd_quatf(ix: q.0, iy: q.1, iz: q.2, r: q.3) : nil
    }

    /// N s on a dynamic part, at a world point — or at its centre of mass
    /// when `worldPoint` is nil: a push without spin.
    public func addImpulse(_ impulse: SIMD3<Float>, toPart index: Int, at worldPoint: SIMD3<Float>? = nil) {
        guard let handle else { return }
        var i: (Float, Float, Float) = (impulse.x, impulse.y, impulse.z)
        guard let worldPoint else {
            withUnsafePointer(to: &i) { ip in
                ip.withMemoryRebound(to: Float.self, capacity: 3) { ujolt_ragdoll_add_impulse(handle, Int32(index), $0, nil) }
            }
            return
        }
        var p: (Float, Float, Float) = (worldPoint.x, worldPoint.y, worldPoint.z)
        withUnsafePointer(to: &i) { ip in
            withUnsafePointer(to: &p) { pp in
                ip.withMemoryRebound(to: Float.self, capacity: 3) { iF in
                    pp.withMemoryRebound(to: Float.self, capacity: 3) { pF in
                        ujolt_ragdoll_add_impulse(handle, Int32(index), iF, pF)
                    }
                }
            }
        }
    }

    /// Flattens one transform per part into the scratch, column-major, and
    /// hands it over; does nothing for any other count.
    private func withFlattened(_ transforms: [simd_float4x4], _ body: (UnsafePointer<Float>) -> Void) {
        guard transforms.count == partCount else { return }
        for (index, transform) in transforms.enumerated() {
            let base = index * 16
            for column in 0 ..< 4 {
                let c = transform[column]
                poseScratch[base + column * 4] = c.x
                poseScratch[base + column * 4 + 1] = c.y
                poseScratch[base + column * 4 + 2] = c.z
                poseScratch[base + column * 4 + 3] = c.w
            }
        }
        poseScratch.withUnsafeBufferPointer { body($0.baseAddress!) }
    }

    func invalidate() {
        handle = nil
    }
}

extension JoltPhysicsBackend {
    /// Adds a ragdoll; nil for a malformed one (no parts, a root that is not
    /// part 0 or more than one, a parent after its child, a convex hull) or
    /// when Jolt is out of bodies. Frame thread.
    public func addRagdoll(_ descriptor: JoltRagdollDescriptor) -> JoltRagdoll? {
        guard !descriptor.parts.isEmpty else { return nil }
        // Jolt copies the names into its skeleton: the C strings only have
        // to outlive the call.
        let names = descriptor.parts.map { strdup($0.name) }
        defer { for name in names { free(name) } }
        var parts: [ujolt_ragdoll_part] = []
        parts.reserveCapacity(descriptor.parts.count)
        for (index, part) in descriptor.parts.enumerated() {
            var c = ujolt_ragdoll_part()
            c.name = UnsafePointer(names[index])
            c.parent = part.parentIndex.map { Int32($0) } ?? -1
            switch part.shape {
            case let .sphere(radius):
                c.shape = UJOLT_SHAPE_SPHERE
                c.radius = radius
            case let .box(halfExtents):
                c.shape = UJOLT_SHAPE_BOX
                c.half_extents = (halfExtents.x, halfExtents.y, halfExtents.z)
            case let .capsule(radius, height):
                c.shape = UJOLT_SHAPE_CAPSULE
                c.radius = radius
                c.half_height = height * 0.5
            case let .cylinder(radius, height):
                c.shape = UJOLT_SHAPE_CYLINDER
                c.radius = radius
                c.half_height = height * 0.5
            case .convexHull:
                return nil
            }
            c.shape_offset = (part.shapeOffset.x, part.shapeOffset.y, part.shapeOffset.z)
            let s = part.shapeRotation
            c.shape_rotation = (s.imag.x, s.imag.y, s.imag.z, s.real)
            c.mass = part.mass
            c.position = (part.position.x, part.position.y, part.position.z)
            let r = part.rotation
            c.rotation = (r.imag.x, r.imag.y, r.imag.z, r.real)
            c.twist_axis = (part.twistAxis.x, part.twistAxis.y, part.twistAxis.z)
            c.plane_axis = (part.planeAxis.x, part.planeAxis.y, part.planeAxis.z)
            let parentTwist = part.parentTwistAxis ?? .zero
            let parentPlane = part.parentPlaneAxis ?? .zero
            c.parent_twist_axis = (parentTwist.x, parentTwist.y, parentTwist.z)
            c.parent_plane_axis = (parentPlane.x, parentPlane.y, parentPlane.z)
            c.twist_min_deg = part.twistRange.lowerBound * 180 / .pi
            c.twist_max_deg = part.twistRange.upperBound * 180 / .pi
            c.normal_half_cone_deg = part.normalHalfConeAngle * 180 / .pi
            c.plane_half_cone_deg = part.planeHalfConeAngle * 180 / .pi
            c.motor_frequency = part.motorFrequency
            c.motor_damping = part.motorDamping
            c.max_torque = part.maxTorque
            c.friction_torque = part.frictionTorque
            c.friction = part.friction
            parts.append(c)
        }
        var desc = ujolt_ragdoll_desc()
        desc.part_count = UInt32(parts.count)
        desc.user_data = UInt64(descriptor.entity)
        desc.layer = descriptor.layer
        desc.gravity_factor = descriptor.gravityFactor
        desc.linear_damping = descriptor.linearDamping
        desc.angular_damping = descriptor.angularDamping
        desc.max_linear_velocity = descriptor.maxLinearVelocity
        desc.start_active = descriptor.startActive ? 1 : 0
        desc.velocity_steps = UInt32(max(descriptor.solverVelocitySteps, 0))
        desc.position_steps = UInt32(max(descriptor.solverPositionSteps, 0))
        let flatPairs = descriptor.disabledCollisionPairs.flatMap { [Int32($0.0), Int32($0.1)] }
        let handle: OpaquePointer? = flatPairs.withUnsafeBufferPointer { pairBuffer in
            desc.disabled_pairs = pairBuffer.baseAddress
            desc.disabled_pair_count = UInt32(descriptor.disabledCollisionPairs.count)
            return parts.withUnsafeBufferPointer { buffer in
            desc.parts = buffer.baseAddress
            return ujolt_world_add_ragdoll(worldHandle, &desc)
            }
        }
        guard let handle else { return nil }
        return JoltRagdoll(handle: handle, descriptor: descriptor)
    }

    /// Removes the ragdoll and its parts. Frame thread.
    public func removeRagdoll(_ ragdoll: JoltRagdoll) {
        guard let handle = ragdoll.handle else { return }
        ujolt_world_remove_ragdoll(worldHandle, handle)
        ragdoll.invalidate()
    }
}
