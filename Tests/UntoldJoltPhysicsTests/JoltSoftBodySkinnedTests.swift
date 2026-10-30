//
//  JoltSoftBodySkinnedTests.swift
//  UntoldJoltPhysics
//

import Foundation
import simd
import UntoldEngine
@testable import UntoldJoltPhysics
import XCTest

/// A sheet skinned to two joints: hard skinning, a step before the first
/// feed, following moving joints, the back-stop, hard-skinned vertices
/// inside a collider, teleport resets, malformed skins, one-way contact
/// against a ragdoll part, the skin parameters, the normals read-back,
/// sleep control, a body put to sleep and woken, and the demo's 100 Hz
/// cadence.
final class JoltSoftBodySkinnedTests: XCTestCase {
    private let step: Float = 1.0 / 60.0
    private let step100: Float = 1.0 / 100.0

    /// A world with a static floor whose top is y = 0.
    private func makeBackend(collisionSteps: Int32 = 1) -> JoltPhysicsBackend {
        var settings = JoltWorldSettings()
        settings.workerThreads = 0
        settings.collisionSteps = collisionSteps
        let backend = JoltPhysicsBackend(settings: settings)
        backend.configure(PhysicsWorldConfiguration())
        backend.didAddBody(entity: 1000, descriptor: PhysicsBodyDescriptor(
            motionType: .static,
            collider: PhysicsColliderDescriptor(shape: .box(halfExtents: simd_float3(50, 0.5, 50)), friction: 0.5, restitution: 0.0),
            position: simd_float3(0, -0.5, 0)
        ))
        return backend
    }

    /// Steps at 60 Hz for `seconds`, calling `eachFrame` before every step
    /// the way a game frame does.
    private func advance(_ backend: JoltPhysicsBackend, seconds: Float, eachFrame: (() -> Void)? = nil) {
        for _ in 0 ..< Int((seconds / step).rounded()) {
            eachFrame?()
            backend.step(deltaTime: step)
        }
    }

    /// The demo's cadence: 100 Hz steps (on a world made with three Jolt
    /// sub-steps). `eachFrame` gets the time at the end of the coming step,
    /// `afterStep` runs once the step is done.
    private func advance100Hz(_ backend: JoltPhysicsBackend, seconds: Float, eachFrame: ((Float) -> Void)? = nil, afterStep: (() -> Void)? = nil) {
        for frame in 0 ..< Int((seconds / step100).rounded()) {
            eachFrame?(Float(frame + 1) * step100)
            backend.step(deltaTime: step100)
            afterStep?()
        }
    }

    // MARK: - The skinned sheet

    /// Columns along x, rows along z: a 10×20 sheet of 5 cm cells lying in
    /// the XZ plane at y 1.5 (the body's frame sits there and the vertices
    /// are authored relative to it), skinned to two joints at x ∓0.25 on
    /// its first row. Every vertex is pulled by both joints by inverse
    /// distance (Jolt's sample recipe); row 0 is hard skinned, the rest may
    /// stray `maxDistance` from the skin and no closer than `backstop`
    /// behind it.
    private let columns = 10
    private let rows = 20
    private let spacing: Float = 0.05
    private let origin = SIMD3<Float>(0, 1.5, 0)
    private let jointBindPositions = [SIMD3<Float>(-0.25, 1.5, 0), SIMD3<Float>(0.25, 1.5, 0)]
    private let maxDistance: Float = 0.05
    private let backstop: Float = 0.005

    private func index(_ row: Int, _ column: Int) -> Int { row * columns + column }

    /// The vertex's authored (bind) position, world.
    private func bindPosition(_ row: Int, _ column: Int) -> SIMD3<Float> {
        origin + SIMD3<Float>((Float(column) - Float(columns - 1) * 0.5) * spacing, 0, Float(row) * spacing)
    }

    private func translation(_ t: SIMD3<Float>) -> simd_float4x4 {
        var m = matrix_identity_float4x4
        m.columns.3 = SIMD4<Float>(t, 1)
        return m
    }

    private var bindPose: [simd_float4x4] { jointBindPositions.map(translation) }

    /// The pose with both joints moved by `offset`.
    private func pose(offset: SIMD3<Float>) -> [simd_float4x4] { jointBindPositions.map { translation($0 + offset) } }

    private func influences(_ row: Int, _ column: Int) -> [JoltSoftBodySkinInfluence] {
        let p = bindPosition(row, column)
        return jointBindPositions.enumerated().map { joint, position in
            JoltSoftBodySkinInfluence(joint: UInt32(joint), weight: 1 / simd_distance(p, position))
        }
    }

    /// Where the skin puts every vertex for `pose`: the bridge's blend,
    /// done on the CPU.
    private func cpuSkin(_ pose: [simd_float4x4]) -> [SIMD3<Float>] {
        let skinMatrices = zip(pose, bindPose).map { $0 * $1.inverse }
        var targets: [SIMD3<Float>] = []
        targets.reserveCapacity(rows * columns)
        for row in 0 ..< rows {
            for column in 0 ..< columns {
                let p = SIMD4<Float>(bindPosition(row, column), 1)
                let weights = influences(row, column)
                let total = weights.reduce(Float(0)) { $0 + $1.weight }
                var skinned = SIMD3<Float>.zero
                for w in weights {
                    let moved = skinMatrices[Int(w.joint)] * p
                    skinned += (w.weight / total) * SIMD3<Float>(moved.x, moved.y, moved.z)
                }
                targets.append(skinned)
            }
        }
        return targets
    }

    /// The sheet: constraints from its triangles (stretch 1e-6, shear 1e-5,
    /// distance bends 1e-3), three iterations, never sleeping unless asked.
    private func sheet(maxDistance: Float? = nil, backstopDistance: Float? = nil, allowSleeping: Bool = false,
                       gravityFactor: Float = 1, oneWayAgainstRagdolls: Bool = false) -> JoltSoftBodyDescriptor
    {
        var vertices: [SIMD3<Float>] = []
        var inverseMasses: [Float] = []
        var skinned: [JoltSoftBodySkinnedVertex] = []
        for row in 0 ..< rows {
            for column in 0 ..< columns {
                vertices.append(bindPosition(row, column) - origin)
                inverseMasses.append(1 / 0.02)
                var vertex = JoltSoftBodySkinnedVertex(
                    vertex: UInt32(index(row, column)), influences: influences(row, column),
                    maxDistance: row == 0 ? 0 : (maxDistance ?? self.maxDistance)
                )
                vertex.backstopDistance = backstopDistance ?? backstop
                skinned.append(vertex)
            }
        }
        var faces: [SIMD3<UInt32>] = []
        for row in 0 ..< rows - 1 {
            for column in 0 ..< columns - 1 {
                // Wound so the normals point +Y.
                let a = UInt32(index(row, column)), b = UInt32(index(row + 1, column))
                let c = UInt32(index(row, column + 1)), d = UInt32(index(row + 1, column + 1))
                faces.append(SIMD3(a, b, c))
                faces.append(SIMD3(b, d, c))
            }
        }
        var descriptor = JoltSoftBodyDescriptor(
            vertices: vertices, inverseMasses: inverseMasses, faces: faces,
            compliance: 1e-6, shearCompliance: 1e-5, bendCompliance: 1e-3
        )
        descriptor.bendType = .distance
        descriptor.iterations = 3
        descriptor.position = origin
        descriptor.vertexRadius = 0.01
        descriptor.allowSleeping = allowSleeping
        descriptor.gravityFactor = gravityFactor
        descriptor.oneWayAgainstRagdolls = oneWayAgainstRagdolls
        descriptor.skin = JoltSoftBodySkin(bindJointWorldMatrices: bindPose, vertices: skinned)
        return descriptor
    }

    private func positions(_ backend: JoltPhysicsBackend, _ body: JoltSoftBody) -> [SIMD3<Float>] {
        var p: [SIMD3<Float>] = []
        XCTAssertEqual(backend.readSoftBodyVertices(body, into: &p), body.vertexCount)
        return p
    }

    /// The largest distance between corresponding entries, over the sheet's
    /// `rows` (all of it when nil).
    private func farthest(_ a: [SIMD3<Float>], from b: [SIMD3<Float>], rows range: Range<Int>? = nil) -> Float {
        let indices = range.map { Array($0.lowerBound * columns ..< $0.upperBound * columns) } ?? Array(0 ..< a.count)
        return indices.reduce(Float(0)) { max($0, simd_distance(a[$1], b[$1])) }
    }

    private func allFinite(_ p: [SIMD3<Float>]) -> Bool {
        p.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }
    }

    // MARK: - Tests

    func testHardSkinPlacesEveryVertexOnItsJoints() throws {
        let backend = makeBackend()
        let body = try XCTUnwrap(backend.addSoftBody(sheet(gravityFactor: 0)))
        XCTAssertTrue(body.isSkinned)
        XCTAssertEqual(body.jointCount, 2)
        // The bind pose first (where the vertices already are), then a hard
        // skin to a pose that is not the bind pose, so the placement shows:
        // both joints shifted and lifted.
        XCTAssertTrue(backend.skinSoftBody(body, jointWorldMatrices: bindPose))
        backend.step(deltaTime: step)
        let shifted = pose(offset: SIMD3<Float>(0.1, 0.05, -0.02))
        let targets = cpuSkin(shifted)
        XCTAssertFalse(backend.skinSoftBody(body, jointWorldMatrices: [shifted[0]], hardSkinAll: true), "one joint is not the skin's two")
        XCTAssertTrue(backend.skinSoftBody(body, jointWorldMatrices: shifted, hardSkinAll: true))
        var p = positions(backend, body)
        XCTAssertLessThan(farthest(p, from: targets), 1e-4, "every vertex sits on its skinned position")
        backend.step(deltaTime: step)
        p = positions(backend, body)
        XCTAssertTrue(allFinite(p))
        XCTAssertLessThan(farthest(p, from: targets), 1e-3, "and stays there through a step without gravity")
    }

    /// Jolt blends the skinned positions from the previous feed's over its
    /// iterations; before the first feed there is none, so the bridge makes
    /// the first feed a hard skin whatever was asked, or the sheet would
    /// dissolve into NaNs on its first step.
    func testFirstFeedIsAHardSkinWhateverIsAsked() throws {
        let backend = makeBackend()
        let body = try XCTUnwrap(backend.addSoftBody(sheet(gravityFactor: 0)))
        let shifted = pose(offset: SIMD3<Float>(0.1, 0.05, -0.02))
        XCTAssertTrue(backend.skinSoftBody(body, jointWorldMatrices: shifted))
        backend.step(deltaTime: step)
        let p = positions(backend, body)
        XCTAssertTrue(allFinite(p))
        XCTAssertLessThan(farthest(p, from: cpuSkin(shifted)), 1e-3, "the first feed placed every vertex on the pose")
    }

    /// Jolt starts the skin state as NaN and runs the skin constraints from
    /// the body's first step, so a body stepped before its first feed — one
    /// added after this frame's `willStep`, say — would dissolve unless the
    /// bridge created it skinned to its bind pose. That priming must not
    /// count as the first feed, which stays a hard skin.
    func testSkinnedBodySurvivesAStepBeforeItsFirstFeed() throws {
        let backend = makeBackend()
        let body = try XCTUnwrap(backend.addSoftBody(sheet(gravityFactor: 0)))
        backend.step(deltaTime: step)
        var p = positions(backend, body)
        XCTAssertTrue(allFinite(p), "an unfed body is still a body")
        let bind = cpuSkin(bindPose)
        XCTAssertLessThan(farthest(p, from: bind, rows: 0 ..< 1), 1e-4, "the hard row sits on its bind pose")
        XCTAssertLessThan(farthest(p, from: bind), 1e-3, "and so, at rest without gravity, does the rest")
        let shifted = pose(offset: SIMD3<Float>(0.1, 0.05, -0.02))
        XCTAssertTrue(backend.skinSoftBody(body, jointWorldMatrices: shifted))
        backend.step(deltaTime: step)
        p = positions(backend, body)
        XCTAssertTrue(allFinite(p))
        XCTAssertLessThan(farthest(p, from: cpuSkin(shifted)), 1e-3, "the first real feed is still a hard skin")
    }

    func testSkinnedSheetFollowsMovingJoints() throws {
        let backend = makeBackend()
        let body = try XCTUnwrap(backend.addSoftBody(sheet()))
        XCTAssertTrue(backend.skinSoftBody(body, jointWorldMatrices: bindPose, hardSkinAll: true))
        // The joints slide 30 cm along x over a second, fed from the hook a
        // game would use.
        var current = bindPose
        var deltas: [Float] = []
        backend.willStep = { [unowned backend, unowned self] dt in
            deltas.append(dt)
            current = self.pose(offset: SIMD3<Float>(0.3 * Float(deltas.count) / 60, 0, 0))
            XCTAssertTrue(backend.skinSoftBody(body, jointWorldMatrices: current))
        }
        advance(backend, seconds: 1)
        backend.willStep = nil
        XCTAssertEqual(deltas.count, 60)
        XCTAssertEqual(deltas.first, step)
        let targets = cpuSkin(current)
        XCTAssertEqual(targets[index(0, 0)].x, bindPosition(0, 0).x + 0.3, accuracy: 1e-5)
        let p = positions(backend, body)
        XCTAssertTrue(allFinite(p))
        XCTAssertLessThan(farthest(p, from: targets), maxDistance + 0.001, "every vertex came along within its slack")
        XCTAssertLessThan(farthest(p, from: targets, rows: 0 ..< 1), 1e-4, "the hard row rides exactly on its joints")
    }

    func testBackstopKeepsClothOutsideTheSkinnedSurface() {
        // Slack enough (50 cm) that only the back-stop can hold the sheet up.
        func lowestPoint(backstopDistance: Float) -> Float {
            let backend = makeBackend()
            let body = backend.addSoftBody(sheet(maxDistance: 0.5, backstopDistance: backstopDistance))!
            backend.skinSoftBody(body, jointWorldMatrices: bindPose, hardSkinAll: true)
            advance(backend, seconds: 2) { backend.skinSoftBody(body, jointWorldMatrices: self.bindPose) }
            let p = positions(backend, body)
            XCTAssertTrue(allFinite(p))
            return p.map(\.y).min()!
        }
        XCTAssertGreaterThanOrEqual(lowestPoint(backstopDistance: backstop), 1.5 - 0.006, "the back-stop keeps the sheet at the skinned surface")
        XCTAssertLessThan(lowestPoint(backstopDistance: .greatestFiniteMagnitude), 1.4, "without it the sheet hangs")
    }

    func testHardSkinnedVerticesIgnoreCollidersWithoutJitter() throws {
        let backend = makeBackend()
        // A static slab whose +z face is 2 cm past row 0 (z 0), so the hard
        // row sits inside it; row 1 (z 0.05) is clear of it.
        backend.didAddBody(entity: 2, descriptor: PhysicsBodyDescriptor(
            motionType: .static,
            collider: PhysicsColliderDescriptor(shape: .box(halfExtents: simd_float3(0.5, 0.1, 0.05)), friction: 0.5, restitution: 0.0),
            position: simd_float3(0, 1.5, -0.03)
        ))
        let body = try XCTUnwrap(backend.addSoftBody(sheet()))
        backend.skinSoftBody(body, jointWorldMatrices: bindPose, hardSkinAll: true)
        let targets = cpuSkin(bindPose)
        // Let the free rows settle onto the back-stop first.
        advance(backend, seconds: 0.5) { backend.skinSoftBody(body, jointWorldMatrices: self.bindPose) }
        var previous = positions(backend, body)
        var worstRow0: Float = 0
        var worstRow1Jump: Float = 0
        for _ in 0 ..< 60 {
            backend.skinSoftBody(body, jointWorldMatrices: bindPose)
            backend.step(deltaTime: step)
            let p = positions(backend, body)
            worstRow0 = max(worstRow0, farthest(p, from: targets, rows: 0 ..< 1))
            worstRow1Jump = max(worstRow1Jump, farthest(p, from: previous, rows: 1 ..< 2))
            previous = p
        }
        XCTAssertLessThan(worstRow0, 1e-4, "the hard row ignores the slab it sits in: off by \(worstRow0) m")
        XCTAssertLessThan(worstRow1Jump, 0.002, "and the row next to it is not jerked about: \(worstRow1Jump) m in a frame")
    }

    func testTeleportThenHardSkinResets() throws {
        let backend = makeBackend()
        let body = try XCTUnwrap(backend.addSoftBody(sheet(gravityFactor: 0)))
        backend.skinSoftBody(body, jointWorldMatrices: bindPose, hardSkinAll: true)
        advance(backend, seconds: 0.5) { backend.skinSoftBody(body, jointWorldMatrices: self.bindPose) }
        let far = pose(offset: SIMD3<Float>(2, 0, 0))
        let targets = cpuSkin(far)
        // A plain feed after a 2 m jump only drags the free rows to the
        // edge of their slack.
        XCTAssertTrue(backend.skinSoftBody(body, jointWorldMatrices: far))
        backend.step(deltaTime: step)
        XCTAssertGreaterThan(farthest(positions(backend, body), from: targets, rows: 1 ..< rows), 0.04, "a soft feed leaves the free rows trailing")
        // A hard skin puts everything on the new pose at rest.
        XCTAssertTrue(backend.skinSoftBody(body, jointWorldMatrices: far, hardSkinAll: true))
        backend.step(deltaTime: step)
        let p = positions(backend, body)
        XCTAssertTrue(allFinite(p))
        XCTAssertLessThan(farthest(p, from: targets), 0.001, "every vertex is on the new pose")
    }

    /// Two triangles at the fixture's height, wound for +Y, the given
    /// vertices skinned to one joint with 5 cm of slack and the given
    /// back-stop.
    private func twoTriangles(skinned: [UInt32], backstop: Float) -> JoltSoftBodyDescriptor {
        var descriptor = JoltSoftBodyDescriptor(
            vertices: [SIMD3<Float>(0, 0, 0), SIMD3<Float>(0.05, 0, 0), SIMD3<Float>(0, 0, 0.05), SIMD3<Float>(0.05, 0, 0.05)],
            inverseMasses: [50, 50, 50, 50], faces: [SIMD3(0, 2, 1), SIMD3(1, 2, 3)],
            compliance: 1e-6, shearCompliance: 1e-5, bendCompliance: 1e-3
        )
        descriptor.position = origin
        descriptor.skin = JoltSoftBodySkin(bindJointWorldMatrices: [bindPose[0]], vertices: skinned.map { vertex in
            var v = JoltSoftBodySkinnedVertex(vertex: vertex, influences: [JoltSoftBodySkinInfluence(joint: 0, weight: 1)], maxDistance: 0.05)
            v.backstopDistance = backstop
            return v
        })
        return descriptor
    }

    /// The documented refusals: a skin without faces, a vertex pulled by no
    /// joint, by five or by a joint the skin does not have, a back-stop on
    /// a vertex with no face whose three vertices are all skinned, a
    /// multiplier that is not finite and a feed with a matrix that is not.
    func testMalformedSkinIsRefused() throws {
        let backend = makeBackend()
        var noFaces = sheet()
        noFaces.faces = []
        XCTAssertNil(backend.addSoftBody(noFaces), "a skin needs faces")
        var noJoint = sheet()
        noJoint.skin!.vertices[5].influences = []
        XCTAssertNil(backend.addSoftBody(noJoint), "a vertex pulled by no joint")
        var fiveJoints = sheet()
        fiveJoints.skin!.vertices[5].influences = Array(repeating: JoltSoftBodySkinInfluence(joint: 0, weight: 0.2), count: 5)
        XCTAssertNil(backend.addSoftBody(fiveJoints), "Jolt takes four joints at most")
        var unknownJoint = sheet()
        unknownJoint.skin!.vertices[5].influences[0].joint = 2
        XCTAssertNil(backend.addSoftBody(unknownJoint), "the skin has two joints")

        // No face of the pair has three skinned vertices, so a back-stop on
        // either skinned vertex has no normal to lie behind.
        XCTAssertNil(backend.addSoftBody(twoTriangles(skinned: [0, 1], backstop: 0.01)), "a back-stop needs a fully skinned face")
        XCTAssertNotNil(backend.addSoftBody(twoTriangles(skinned: [0, 1], backstop: .greatestFiniteMagnitude)), "without the back-stop the same skin is fine")
        XCTAssertNotNil(backend.addSoftBody(twoTriangles(skinned: [0, 1, 2, 3], backstop: 0.01)), "as is the back-stop once a face is fully skinned")

        // A NaN multiplier applies nothing: std::max would have made it 0
        // and hard skinned the whole sheet onto its skin.
        let hanging = try XCTUnwrap(backend.addSoftBody(sheet()))
        backend.skinSoftBody(hanging, jointWorldMatrices: bindPose, hardSkinAll: true)
        advance(backend, seconds: 0.5) { backend.skinSoftBody(hanging, jointWorldMatrices: self.bindPose) }
        backend.setSoftBodySkin(hanging, maxDistanceMultiplier: .nan)
        backend.skinSoftBody(hanging, jointWorldMatrices: bindPose)
        backend.step(deltaTime: step)
        XCTAssertGreaterThan(farthest(positions(backend, hanging), from: cpuSkin(bindPose), rows: 1 ..< rows), 0.003, "the free rows still hang on the back-stop")

        // A feed with a NaN in it applies nothing and the body is unharmed.
        let body = try XCTUnwrap(backend.addSoftBody(sheet(gravityFactor: 0)))
        var bad = bindPose
        bad[1].columns.3.x = .nan
        XCTAssertFalse(backend.skinSoftBody(body, jointWorldMatrices: bad), "a non-finite matrix is refused")
        backend.step(deltaTime: step)
        let p = positions(backend, body)
        XCTAssertTrue(allFinite(p))
        XCTAssertLessThan(farthest(p, from: cpuSkin(bindPose)), 1e-3, "and nothing of it was applied")
    }

    // MARK: Ragdoll parts

    /// One free capsule part along +Y — the pelvis of JoltRagdollTests'
    /// chain — whose dome tops out at y 1.35.
    private func lonePart() -> JoltRagdollDescriptor {
        var part = JoltRagdollPart(
            name: "pelvis", parentIndex: nil, shape: .capsule(radius: 0.1, height: 0.2), mass: 10,
            position: SIMD3<Float>(0, 1.0, 0), rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1),
            twistAxis: SIMD3<Float>(0, 1, 0), planeAxis: SIMD3<Float>(1, 0, 0)
        )
        part.shapeOffset = SIMD3<Float>(0, 0.15, 0)
        var descriptor = JoltRagdollDescriptor(parts: [part])
        descriptor.entity = 42
        descriptor.startActive = true
        // Coasting is what is measured: Jolt's 5 %/s damping would cost
        // 1.2 cm of the half metre on its own.
        descriptor.linearDamping = 1e-4
        return descriptor
    }

    /// A 7×7 sheet of `particleMass` kg particles, 10 cm cells, at height
    /// `y`, triangulated; its corners pinned when asked.
    private func plainSheet(y: Float, particleMass: Float, pinCorners: Bool, oneWay: Bool) -> JoltSoftBodyDescriptor {
        let n = 7
        let spacing: Float = 0.1
        var vertices: [SIMD3<Float>] = []
        var inverseMasses: [Float] = []
        let half = Float(n - 1) * spacing * 0.5
        for row in 0 ..< n {
            for column in 0 ..< n {
                vertices.append(SIMD3<Float>(Float(column) * spacing - half, y, Float(row) * spacing - half))
                let corner = (row == 0 || row == n - 1) && (column == 0 || column == n - 1)
                inverseMasses.append(pinCorners && corner ? 0 : 1 / particleMass)
            }
        }
        var edges: [SIMD2<UInt32>] = []
        var faces: [SIMD3<UInt32>] = []
        func index(_ row: Int, _ column: Int) -> UInt32 { UInt32(row * n + column) }
        for row in 0 ..< n {
            for column in 0 ..< n {
                if column + 1 < n { edges.append(SIMD2(index(row, column), index(row, column + 1))) }
                if row + 1 < n { edges.append(SIMD2(index(row, column), index(row + 1, column))) }
                if row + 1 < n, column + 1 < n {
                    edges.append(SIMD2(index(row, column), index(row + 1, column + 1)))
                    faces.append(SIMD3(index(row, column), index(row + 1, column), index(row, column + 1)))
                    faces.append(SIMD3(index(row + 1, column), index(row + 1, column + 1), index(row, column + 1)))
                }
            }
        }
        var descriptor = JoltSoftBodyDescriptor(vertices: vertices, inverseMasses: inverseMasses, edges: edges)
        descriptor.faces = faces
        descriptor.compliance = 1e-6
        descriptor.iterations = 10
        descriptor.vertexRadius = 0.03
        descriptor.allowSleeping = false
        descriptor.oneWayAgainstRagdolls = oneWay
        return descriptor
    }

    private func partX(_ ragdoll: JoltRagdoll) -> Float {
        var pose: [simd_float4x4] = []
        ragdoll.readPose(into: &pose)
        return pose[0].columns.3.x
    }

    func testRagdollPartsPushTheClothButTheClothDoesNotPushThem() throws {
        /// Drapes a heavy (98 kg) free sheet over the part, then sends the
        /// part off at 0.5 m/s along x and lets it coast for a second: how
        /// far it got, and how far the sheet's centre was carried. The
        /// velocity is set once because Jolt applies a soft body's impulses
        /// after the rigid bodies have integrated; a velocity re-imposed
        /// before every step would erase them, two-way or not.
        func drive(oneWay: Bool) throws -> (part: Float, cloth: Float) {
            let backend = makeBackend()
            let ragdoll = try XCTUnwrap(backend.addRagdoll(lonePart()))
            ragdoll.setAllowSleeping(false)
            let cloth = try XCTUnwrap(backend.addSoftBody(plainSheet(y: 1.37, particleMass: 2, pinCorners: false, oneWay: oneWay)))
            // The part holds still (kinematic) while the sheet lands on it.
            advance(backend, seconds: 1)
            let centreBefore = positions(backend, cloth)[24]
            ragdoll.setPartDynamic(0, true)
            ragdoll.setGravityFactor(nil, 0)
            let startX = partX(ragdoll)
            ragdoll.setVelocities(linear: [SIMD3<Float>(0.5, 0, 0)], angular: nil)
            advance(backend, seconds: 1)
            let p = positions(backend, cloth)
            XCTAssertTrue(allFinite(p))
            return (partX(ragdoll) - startX, p[24].x - centreBefore.x)
        }
        let oneWay = try drive(oneWay: true)
        XCTAssertEqual(oneWay.part, 0.5, accuracy: 0.002, "the one-way cloth never slows the part: it travelled \(oneWay.part) m")
        XCTAssertGreaterThanOrEqual(oneWay.cloth, 0.02, "yet the part drags the cloth along: \(oneWay.cloth) m")
        let twoWay = try drive(oneWay: false)
        XCTAssertLessThan(twoWay.part, 0.5 - 0.01, "the same sheet pushing back slows the part: it travelled \(twoWay.part) m")
        print("one-way cloth: part travelled \(oneWay.part) m, cloth centre \(oneWay.cloth) m; two-way: part travelled \(twoWay.part) m, cloth centre \(twoWay.cloth) m")

        // One-way is about ragdoll parts only: a plain dynamic body is still
        // caught, as in JoltSoftBodyTests.testSheetCatchesADroppedBall.
        let backend = makeBackend()
        let sheet = try XCTUnwrap(backend.addSoftBody(plainSheet(y: 1.0, particleMass: 0.05, pinCorners: true, oneWay: true)))
        advance(backend, seconds: 0.5)
        backend.didAddBody(entity: 7, descriptor: PhysicsBodyDescriptor(
            motionType: .dynamic,
            collider: PhysicsColliderDescriptor(shape: .sphere(radius: 0.06), friction: 0.4, restitution: 0.1),
            mass: 0.2,
            position: simd_float3(0, 1.6, 0)
        ))
        advance(backend, seconds: 2)
        let ball = try XCTUnwrap(backend.bodyState(for: 7))
        XCTAssertGreaterThan(ball.position.y, 0.5, "the one-way sheet still holds a ball up")
        XCTAssertLessThan(positions(backend, sheet)[24].y, 0.98, "and sags under it")
    }

    // MARK: Parameters, read-back, sleep, cadence

    func testSkinParamsHardSkinAndRelease() throws {
        let backend = makeBackend()
        let body = try XCTUnwrap(backend.addSoftBody(sheet()))
        backend.skinSoftBody(body, jointWorldMatrices: bindPose, hardSkinAll: true)
        let targets = cpuSkin(bindPose)
        // Settled, the free rows hang on the back-stop, 5 mm below their targets.
        advance(backend, seconds: 0.5) { backend.skinSoftBody(body, jointWorldMatrices: self.bindPose) }
        XCTAssertGreaterThan(farthest(positions(backend, body), from: targets, rows: 1 ..< rows), 0.003)
        // A zero multiplier hard skins everything within a step.
        backend.setSoftBodySkin(body, maxDistanceMultiplier: 0)
        backend.skinSoftBody(body, jointWorldMatrices: bindPose)
        backend.step(deltaTime: step)
        XCTAssertLessThan(farthest(positions(backend, body), from: targets), 1e-3, "every vertex is pulled onto its skin")
        // Switched off, only the hard row still follows the skin: joints
        // moved 10 cm take it along (Jolt hard skins the kinematic vertices
        // at feed time when it will not solve the constraints) and leave
        // the rest hanging free.
        backend.setSoftBodySkin(body, maxDistanceMultiplier: 1, enabled: false)
        let moved = pose(offset: SIMD3<Float>(0.1, 0, 0))
        let movedTargets = cpuSkin(moved)
        advance(backend, seconds: 1) { backend.skinSoftBody(body, jointWorldMatrices: moved) }
        let p = positions(backend, body)
        XCTAssertTrue(allFinite(p))
        XCTAssertLessThan(farthest(p, from: movedTargets, rows: 0 ..< 1), 1e-4, "the hard row follows the joints without the constraints")
        XCTAssertGreaterThan(farthest(p, from: movedTargets, rows: 1 ..< rows), maxDistance, "the free rows sag past their slack")
    }

    func testReadbackNormalsAreUnitAndPositionsMatchTheOldCall() throws {
        let backend = makeBackend()
        // Flat and at rest: no gravity, static joints.
        let body = try XCTUnwrap(backend.addSoftBody(sheet(gravityFactor: 0)))
        backend.skinSoftBody(body, jointWorldMatrices: bindPose, hardSkinAll: true)
        advance(backend, seconds: 1) { backend.skinSoftBody(body, jointWorldMatrices: self.bindPose) }
        var old: [SIMD3<Float>] = []
        XCTAssertEqual(backend.readSoftBodyVertices(body, into: &old), rows * columns)
        var new: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        XCTAssertEqual(backend.readSoftBodyVertices(body, into: &new, normals: &normals), rows * columns)
        XCTAssertEqual(normals.count, rows * columns)
        let fiveDegrees = cos(5 * Float.pi / 180)
        for i in 0 ..< old.count {
            XCTAssertEqual(old[i], new[i], "vertex \(i) differs between the two read-backs")
            XCTAssertEqual(simd_length(normals[i]), 1, accuracy: 1e-3, "normal \(i) is unit")
            XCTAssertGreaterThan(normals[i].y, fiveDegrees, "normal \(i) points up: \(normals[i])")
        }
    }

    func testSkinnedBodyForbiddenToSleepStaysActive() {
        func activeAfterRest(allowSleeping: Bool) -> Bool {
            let backend = makeBackend()
            let body = backend.addSoftBody(sheet(allowSleeping: allowSleeping))!
            backend.skinSoftBody(body, jointWorldMatrices: bindPose, hardSkinAll: true)
            advance(backend, seconds: 3)
            return backend.isSoftBodyActive(body)
        }
        XCTAssertFalse(activeAfterRest(allowSleeping: true), "Jolt sleeps a sheet at rest")
        XCTAssertTrue(activeAfterRest(allowSleeping: false), "unless told not to")
    }

    /// A resting sheet told to sleep is left out of the step — not a vertex
    /// moves over a hundred steps, and the world steps in a fraction of the
    /// time it takes with the sheet awake (both printed) — until it is
    /// woken: by the call, by a skin feed or by a skin-parameter change,
    /// after which it follows its joints again.
    func testASheetPutToSleepCostsNothingUntilItIsWoken() throws {
        func settled() throws -> (JoltPhysicsBackend, JoltSoftBody) {
            let backend = makeBackend(collisionSteps: 3)
            let body = try XCTUnwrap(backend.addSoftBody(sheet()))
            backend.skinSoftBody(body, jointWorldMatrices: bindPose, hardSkinAll: true)
            advance(backend, seconds: 1) { backend.skinSoftBody(body, jointWorldMatrices: self.bindPose) }
            return (backend, body)
        }
        /// Microseconds per step over `steps` steps, no feed.
        func stepMicroseconds(_ backend: JoltPhysicsBackend, steps: Int) -> Double {
            let started = DispatchTime.now().uptimeNanoseconds
            for _ in 0 ..< steps { backend.step(deltaTime: step) }
            return Double(DispatchTime.now().uptimeNanoseconds - started) / 1e3 / Double(steps)
        }
        let (resting, sleeper) = try settled()
        XCTAssertTrue(resting.isSoftBodyActive(sleeper), "never told to sleep, the sheet is awake")
        resting.setSoftBodyActive(sleeper, false)
        XCTAssertFalse(resting.isSoftBodyActive(sleeper))
        let before = positions(resting, sleeper)
        let asleepMicroseconds = stepMicroseconds(resting, steps: 100)
        XCTAssertFalse(resting.isSoftBodyActive(sleeper), "nothing in the world woke it")
        XCTAssertEqual(positions(resting, sleeper), before, "asleep, not a vertex moved")

        let (running, _) = try settled()
        let awakeMicroseconds = stepMicroseconds(running, steps: 100)
        print(String(format: "soft body step (%d vertices, 3 iterations, 3 sub-steps, 100 steps): awake %.1f us, asleep %.1f us", rows * columns, awakeMicroseconds, asleepMicroseconds))
        XCTAssertLessThan(asleepMicroseconds, awakeMicroseconds / 4, "a sleeping sheet costs a fraction of an awake one")

        // Woken by the call, it is Jolt's again.
        resting.setSoftBodyActive(sleeper, true)
        XCTAssertTrue(resting.isSoftBodyActive(sleeper))
        // A skin feed wakes it by itself, and the body follows the joints.
        resting.setSoftBodyActive(sleeper, false)
        let moved = pose(offset: SIMD3<Float>(0.1, 0, 0))
        XCTAssertTrue(resting.skinSoftBody(sleeper, jointWorldMatrices: moved))
        XCTAssertTrue(resting.isSoftBodyActive(sleeper), "a skin feed wakes the body")
        advance(resting, seconds: 0.5) { resting.skinSoftBody(sleeper, jointWorldMatrices: moved) }
        let p = positions(resting, sleeper)
        XCTAssertTrue(allFinite(p))
        XCTAssertLessThan(farthest(p, from: cpuSkin(moved), rows: 0 ..< 1), 1e-4, "the hard row followed the joints")
        XCTAssertLessThan(farthest(p, from: cpuSkin(moved), rows: 1 ..< rows), maxDistance + 0.005, "and the rest stayed within its slack")
        // So does a change of the skin parameters.
        resting.setSoftBodyActive(sleeper, false)
        resting.setSoftBodySkin(sleeper, maxDistanceMultiplier: 1)
        XCTAssertTrue(resting.isSoftBodyActive(sleeper), "a skin-parameter change wakes the body")
    }

    /// Jolt wakes a sleeping soft body for any awake rigid body that reaches
    /// its bounds on a layer the two share — the ball thrown at a sleeping
    /// shirt, but just as much a ragdoll part sitting inside it — so a body
    /// to be kept asleep goes on a layer the awake bodies do not meet: there
    /// the ball falls through the sleeping sheet and the sheet sleeps on.
    func testAnAwakeRigidBodyWakesASleepingSheetUnlessTheLayersAreClosed() throws {
        // Layer 1 (the ball's) meets layer 0 (the sheet's); layer 2 meets nothing.
        var masks = Array(repeating: UInt32.max, count: 32)
        masks[2] = 0
        for layer in masks.indices { masks[layer] &= ~(UInt32(1) << 2) }
        func drop(sheetLayer: UInt32) throws -> (woke: Bool, moved: Float, ballHeight: Float) {
            let backend = makeBackend()
            var configuration = PhysicsWorldConfiguration()
            configuration.collisionLayerMatrix = masks
            backend.configure(configuration)
            let body = try XCTUnwrap(backend.addSoftBody(sheet()))
            backend.skinSoftBody(body, jointWorldMatrices: bindPose, hardSkinAll: true)
            advance(backend, seconds: 1) { backend.skinSoftBody(body, jointWorldMatrices: self.bindPose) }
            backend.setSoftBodyLayer(body, sheetLayer)
            backend.setSoftBodyActive(body, false)
            let before = positions(backend, body)
            // A ball above the middle of the sheet, 30 cm up.
            backend.didAddBody(entity: 7, descriptor: PhysicsBodyDescriptor(
                motionType: .dynamic,
                collider: PhysicsColliderDescriptor(shape: .sphere(radius: 0.05), friction: 0.4, restitution: 0.0),
                mass: 0.2, layer: 1,
                position: bindPosition(rows / 2, columns / 2) + SIMD3<Float>(0, 0.3, 0)
            ))
            var woke = false
            advance(backend, seconds: 1.5) { woke = woke || backend.isSoftBodyActive(body) }
            let ball = try XCTUnwrap(backend.bodyState(for: 7))
            return (woke, farthest(positions(backend, body), from: before), ball.position.y)
        }
        let shared = try drop(sheetLayer: 0)
        XCTAssertTrue(shared.woke, "the ball woke the sleeping sheet through Jolt")
        XCTAssertGreaterThan(shared.moved, 0.005, "which then caught it: the sheet moved \(shared.moved) m")
        XCTAssertGreaterThan(shared.ballHeight, origin.y - 0.2, "the ball rests on the sheet at \(shared.ballHeight) m")
        let closed = try drop(sheetLayer: 2)
        XCTAssertFalse(closed.woke, "on a layer the ball does not meet, the sheet slept on")
        XCTAssertEqual(closed.moved, 0, "and not a vertex moved")
        XCTAssertLessThan(closed.ballHeight, 0.2, "the ball fell through to the floor: \(closed.ballHeight) m")
    }

    /// The demo's cadence, one skin feed per step: the joints swing
    /// x = 0.2 sin(2πt) at 100 Hz for 3 s, with three Jolt sub-steps and
    /// with one. Printed for tuning and bounded: the worst vertex loosely,
    /// since Jolt solves the stretch edges after the skin clamp within an
    /// iteration and a fast swing leaves a vertex a few millimetres past
    /// its slack at read-back; the RMS tracking error with a third over
    /// what was measured (19 and 26 mm), so a regression doubling it shows.
    func testOneSkinPerUpdateCadence() throws {
        func run(collisionSteps: Int32) throws -> (rms: Float, peakSpeed: Float, worst: Float) {
            let backend = makeBackend(collisionSteps: collisionSteps)
            let body = try XCTUnwrap(backend.addSoftBody(sheet()))
            backend.skinSoftBody(body, jointWorldMatrices: bindPose, hardSkinAll: true)
            var previous = positions(backend, body)
            var current = bindPose
            var sumSquares: Float = 0
            var samples = 0
            var peakSpeed: Float = 0
            var worst: Float = 0
            advance100Hz(backend, seconds: 3) { t in
                current = self.pose(offset: SIMD3<Float>(0.2 * sin(2 * Float.pi * t), 0, 0))
                XCTAssertTrue(backend.skinSoftBody(body, jointWorldMatrices: current))
            } afterStep: {
                let p = self.positions(backend, body)
                XCTAssertTrue(self.allFinite(p))
                let targets = self.cpuSkin(current)
                for i in 0 ..< p.count {
                    let d = simd_distance(p[i], targets[i])
                    sumSquares += d * d
                    worst = max(worst, d)
                    peakSpeed = max(peakSpeed, simd_distance(p[i], previous[i]) / self.step100)
                }
                samples += p.count
                previous = p
            }
            return (sqrt(sumSquares / Float(samples)), peakSpeed, worst)
        }
        for substeps: Int32 in [3, 1] {
            let result = try run(collisionSteps: substeps)
            XCTAssertLessThan(result.worst, maxDistance + 0.005, "\(substeps) sub-steps: a vertex strayed \(result.worst) m from its skin")
            XCTAssertLessThan(result.rms, 0.035, "\(substeps) sub-steps: RMS vertex-to-target \(result.rms) m")
            print("skin cadence: 100 Hz, \(substeps) sub-step(s) -> RMS vertex-to-target \(result.rms) m, peak particle speed \(result.peakSpeed) m/s")
        }
    }
}
