//
//  JoltRagdollTests.swift
//  UntoldJoltPhysics
//

import simd
import UntoldEngine
@testable import UntoldJoltPhysics
import XCTest

private final class RecordingSink: PhysicsEventSink {
    var contacts: [PhysicsContactEvent] = []
    var triggers: [PhysicsTriggerEvent] = []
    var activations: [PhysicsBodyActivationEvent] = []
    var dropped = 0

    func receiveContact(_ event: PhysicsContactEvent) { contacts.append(event) }
    func receiveTrigger(_ event: PhysicsTriggerEvent) { triggers.append(event) }
    func receiveActivation(_ event: PhysicsBodyActivationEvent) { activations.append(event) }
    func reportDroppedEvents(count: Int) { dropped += count }
}

final class JoltRagdollTests: XCTestCase {
    private let step: Float = 1.0 / 60.0
    private let zombie: EntityID = 42
    private let identity = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    /// The chain's capsule radii and neutral pivot heights, root first.
    private let radii: [Float] = [0.1, 0.06, 0.05]
    private let heights: [Float] = [1.0, 1.3, 1.6]

    /// A world with a static floor whose top is y = 0.
    private func makeBackend() -> JoltPhysicsBackend {
        var settings = JoltWorldSettings()
        settings.workerThreads = 0
        let backend = JoltPhysicsBackend(settings: settings)
        backend.configure(PhysicsWorldConfiguration())
        backend.didAddBody(entity: 1000, descriptor: PhysicsBodyDescriptor(
            motionType: .static,
            collider: PhysicsColliderDescriptor(shape: .box(halfExtents: simd_float3(50, 0.5, 50)), friction: 0.5, restitution: 0.0),
            position: simd_float3(0, -0.5, 0)
        ))
        return backend
    }

    /// The chain of the tests: a pelvis at 1 m with an upper and a lower
    /// part stacked above it, each a capsule along +Y whose base is at the
    /// joint pivot (so the shape sits 15 cm up the part's frame); cones of
    /// 60°, twist ±30°.
    private func chain(masses: [Float] = [10, 3, 2]) -> JoltRagdollDescriptor {
        let names = ["pelvis", "upper", "lower"]
        let shapes: [PhysicsColliderShape] = [
            .capsule(radius: 0.1, height: 0.2),
            .capsule(radius: 0.06, height: 0.3),
            .capsule(radius: 0.05, height: 0.3),
        ]
        var parts: [JoltRagdollPart] = []
        for index in 0 ..< 3 {
            var part = JoltRagdollPart(
                name: names[index],
                parentIndex: index == 0 ? nil : index - 1,
                shape: shapes[index],
                mass: masses[index],
                position: SIMD3<Float>(0, heights[index], 0),
                rotation: identity,
                twistAxis: SIMD3<Float>(0, 1, 0),
                planeAxis: SIMD3<Float>(1, 0, 0)
            )
            part.shapeOffset = SIMD3<Float>(0, 0.15, 0)
            part.twistRange = (-30 * Float.pi / 180) ... (30 * Float.pi / 180)
            part.normalHalfConeAngle = 60 * .pi / 180
            part.planeHalfConeAngle = 60 * .pi / 180
            parts.append(part)
        }
        var descriptor = JoltRagdollDescriptor(parts: parts)
        descriptor.entity = zombie
        return descriptor
    }

    private func makeChain(_ backend: JoltPhysicsBackend, active: Bool = true) -> JoltRagdoll {
        var descriptor = chain()
        descriptor.startActive = active
        let ragdoll = backend.addRagdoll(descriptor)
        XCTAssertNotNil(ragdoll)
        return ragdoll!
    }

    private func transform(_ position: SIMD3<Float>, _ rotation: simd_quatf? = nil) -> simd_float4x4 {
        var m = simd_float4x4(rotation ?? identity)
        m.columns.3 = SIMD4<Float>(position, 1)
        return m
    }

    private func neutralPose() -> [simd_float4x4] {
        heights.map { transform(SIMD3<Float>(0, $0, 0)) }
    }

    /// The chain with the upper and lower parts leaning 20° about Z (the
    /// pelvis stays): the lower's pivot rides on the upper's tilted axis.
    private func leaningPose() -> [simd_float4x4] {
        let lean = simd_quatf(angle: 20 * .pi / 180, axis: SIMD3<Float>(0, 0, 1))
        let upperPivot = SIMD3<Float>(0, 1.3, 0)
        let lowerPivot = upperPivot + lean.act(SIMD3<Float>(0, 0.3, 0))
        return [transform(SIMD3<Float>(0, 1.0, 0)), transform(upperPivot, lean), transform(lowerPivot, lean)]
    }

    private func positions(_ ragdoll: JoltRagdoll) -> [SIMD3<Float>] {
        var pose: [simd_float4x4] = []
        ragdoll.readPose(into: &pose)
        return pose.map { SIMD3<Float>($0.columns.3.x, $0.columns.3.y, $0.columns.3.z) }
    }

    /// Steps the world for `seconds`, calling `eachFrame` before every step
    /// the way a game frame does.
    private func advance(_ backend: JoltPhysicsBackend, seconds: Float, sink: RecordingSink? = nil, eachFrame: (() -> Void)? = nil) {
        for _ in 0 ..< Int((seconds / step).rounded()) {
            eachFrame?()
            backend.step(deltaTime: step)
            if let sink { backend.drainEvents(into: sink) }
        }
    }

    private func readTransforms(_ backend: JoltPhysicsBackend, capacity: Int = 64) -> [EntityID: PhysicsBodyTransform] {
        var entities = [EntityID](repeating: 0, count: capacity)
        var transforms = [PhysicsBodyTransform](
            repeating: PhysicsBodyTransform(position: .zero, orientation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)),
            count: capacity
        )
        var result: [EntityID: PhysicsBodyTransform] = [:]
        entities.withUnsafeMutableBufferPointer { entityBuffer in
            transforms.withUnsafeMutableBufferPointer { transformBuffer in
                let batch = PhysicsTransformReadBatch(entities: entityBuffer, transforms: transformBuffer)
                let written = backend.readActiveTransforms(into: batch)
                for index in 0 ..< written {
                    result[entityBuffer[index]] = transformBuffer[index]
                }
            }
        }
        return result
    }

    // MARK: - Tests

    func testCreationActivationAndRemoval() {
        let backend = makeBackend()
        let before = backend.bodyCount
        let ragdoll = makeChain(backend, active: false)
        XCTAssertEqual(ragdoll.partCount, 3)
        XCTAssertEqual(ragdoll.partNames, ["pelvis", "upper", "lower"])
        XCTAssertEqual(ragdoll.parentIndices, [nil, 0, 1])
        XCTAssertEqual(ragdoll.entity, zombie)
        XCTAssertEqual(backend.bodyCount, before + 3, "every part is listed, in the world or not")
        XCTAssertFalse(ragdoll.isActive)
        XCTAssertFalse(ragdoll.isPartDynamic(0), "parts start kinematic")

        let neutral = positions(ragdoll)
        XCTAssertEqual(neutral.count, 3)
        XCTAssertEqual(neutral[2].y, 1.6, accuracy: 1e-5)
        ragdoll.setActive(true)
        XCTAssertTrue(ragdoll.isActive)
        XCTAssertEqual(backend.bodyCount, before + 3)
        ragdoll.setActive(true) // twice is harmless
        advance(backend, seconds: 0.2)
        XCTAssertEqual(positions(ragdoll)[2].y, 1.6, accuracy: 1e-4, "kinematic parts given no pose stay put")

        backend.removeRagdoll(ragdoll)
        XCTAssertFalse(ragdoll.isValid)
        XCTAssertFalse(ragdoll.isActive)
        XCTAssertEqual(backend.bodyCount, before)
        var pose: [simd_float4x4] = []
        XCTAssertEqual(ragdoll.readPose(into: &pose), 0)
        backend.removeRagdoll(ragdoll) // twice is harmless
        advance(backend, seconds: 0.1)

        // Malformed descriptions are refused.
        var reversed = chain()
        reversed.parts[1].parentIndex = 2
        XCTAssertNil(backend.addRagdoll(reversed), "a parent must come before its child")
        var twoRoots = chain()
        twoRoots.parts[1].parentIndex = nil
        XCTAssertNil(backend.addRagdoll(twoRoots))
        var hull = chain()
        hull.parts[2].shape = .convexHull(vertices: [.zero, SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0), SIMD3<Float>(0, 0, 1)])
        XCTAssertNil(backend.addRagdoll(hull))
        XCTAssertNil(backend.addRagdoll(JoltRagdollDescriptor(parts: [])))
        XCTAssertEqual(backend.bodyCount, before, "a refused ragdoll leaves nothing behind")
    }

    func testKinematicPartsFollowTheKinematicPose() {
        let backend = makeBackend()
        let ragdoll = makeChain(backend)
        let moved = neutralPose().map { transform(SIMD3<Float>(0.5, $0.columns.3.y, 0)) }
        advance(backend, seconds: 1) {
            ragdoll.setKinematicPose(moved)
        }
        let after = positions(ragdoll)
        for index in 0 ..< 3 {
            XCTAssertEqual(after[index].x, 0.5, accuracy: 1e-3, "part \(index) followed the pose")
            XCTAssertEqual(after[index].y, heights[index], accuracy: 1e-3)
        }
        // The pose is kept: further steps without a new one hold it.
        advance(backend, seconds: 0.2)
        XCTAssertEqual(positions(ragdoll)[0].x, 0.5, accuracy: 1e-3)
        XCTAssertNil(readTransforms(backend)[zombie], "ragdoll parts are never read back")
    }

    func testMotorsHoldTheChainAgainstGravity() {
        let backend = makeBackend()
        let ragdoll = makeChain(backend)
        ragdoll.setPartDynamic(1, true)
        ragdoll.setPartDynamic(2, true)
        XCTAssertFalse(ragdoll.isPartDynamic(0))
        XCTAssertTrue(ragdoll.isPartDynamic(1))
        XCTAssertTrue(ragdoll.isPartDynamic(2))
        ragdoll.setMotors(nil, mode: .position)
        // Start leaning, so the motors have to pull the chain back up and
        // then hold it there (a perfectly vertical chain is balanced and
        // would prove nothing).
        ragdoll.setPose(leaningPose())
        XCTAssertGreaterThan(simd_length(positions(ragdoll)[2] - SIMD3<Float>(0, 1.6, 0)), 0.09)

        let neutral = neutralPose()
        var largestDriftAfterSettling: Float = 0
        var frame = 0
        advance(backend, seconds: 2) {
            ragdoll.driveMotors(toward: neutral)
            frame += 1
            if frame > 60 {
                largestDriftAfterSettling = max(largestDriftAfterSettling, simd_length(self.positions(ragdoll)[2] - SIMD3<Float>(0, 1.6, 0)))
            }
        }
        let lower = positions(ragdoll)[2]
        XCTAssertEqual(simd_length(lower - SIMD3<Float>(0, 1.6, 0)), 0, accuracy: 0.05, "the motors hold the lower part at neutral")
        XCTAssertLessThan(largestDriftAfterSettling, 0.05, "and keep it there through the second second")
    }

    func testLimpChainFoldsUnderFrictionWithoutSinking() {
        let backend = makeBackend()
        let ragdoll = makeChain(backend)
        ragdoll.setPartDynamic(1, true)
        ragdoll.setPartDynamic(2, true)
        ragdoll.setMotors(nil, mode: .off, frictionTorque: 2)
        ragdoll.setPose(leaningPose())
        let lowerStart = positions(ragdoll)[2]

        advance(backend, seconds: 2)
        let after = positions(ragdoll)
        XCTAssertLessThan(after[2].y, lowerStart.y - 0.08, "the chain folded over: the lower part came down")
        XCTAssertEqual(after[0].y, 1.0, accuracy: 1e-4, "the kinematic pelvis did not move")
        for index in 0 ..< 3 {
            XCTAssertGreaterThanOrEqual(after[index].y, radii[index] - 0.01, "part \(index) is not below the floor")
        }
    }

    func testKnockdownComesToRestOnTheFloor() {
        let backend = makeBackend()
        let ragdoll = makeChain(backend)
        ragdoll.setPartDynamic(nil, true)
        XCTAssertTrue((0 ..< 3).allSatisfy { ragdoll.isPartDynamic($0) })
        // Lying along +X, half a metre up: each capsule's axis (its local
        // Y) turned onto the world X axis.
        let lying = simd_quatf(angle: -.pi / 2, axis: SIMD3<Float>(0, 0, 1))
        let pose = [
            transform(SIMD3<Float>(0, 0.5, 0), lying),
            transform(SIMD3<Float>(0.3, 0.5, 0), lying),
            transform(SIMD3<Float>(0.6, 0.5, 0), lying),
        ]
        ragdoll.setPose(pose)
        ragdoll.setVelocities(linear: Array(repeating: .zero, count: 3), angular: nil)

        advance(backend, seconds: 2)
        let resting = positions(ragdoll)
        for index in 0 ..< 3 {
            // A part lying flat rests with its origin one radius up, less
            // Jolt's 2 cm penetration slop; one that hangs off the joint of
            // a thicker neighbour rides a little higher, never lower.
            XCTAssertGreaterThan(resting[index].y, radii[index] - 0.025, "part \(index) does not sink into the floor")
            XCTAssertLessThan(resting[index].y, radii[0] + 0.03, "part \(index) is down on the floor")
        }
        advance(backend, seconds: 1)
        let later = positions(ragdoll)
        for index in 0 ..< 3 {
            XCTAssertEqual(later[index].y, resting[index].y, accuracy: 0.01, "part \(index) stays where it rests")
        }
    }

    func testContactsNameTheRagdollsEntityAndItIsNeverReadBack() {
        let backend = makeBackend()
        let ragdoll = makeChain(backend)
        backend.didAddBody(entity: 7, descriptor: PhysicsBodyDescriptor(
            motionType: .dynamic,
            collider: PhysicsColliderDescriptor(shape: .sphere(radius: 0.11), friction: 0.35, restitution: 0.3),
            mass: 0.43,
            position: simd_float3(0, 2.4, 0)
        ))
        let sink = RecordingSink()
        advance(backend, seconds: 1, sink: sink)
        let hits = sink.contacts.filter { $0.phase == .began && ($0.entityA == zombie || $0.entityB == zombie) }
        XCTAssertFalse(hits.isEmpty, "the ball landing on the chain names the ragdoll's entity")
        XCTAssertEqual(hits.first?.entityA, 7, "A is the dynamic ball")
        XCTAssertEqual(hits.first?.entityB, zombie)
        XCTAssertNil(readTransforms(backend)[zombie])

        // Simulated parts are active dynamic bodies, and still not read back.
        ragdoll.setPartDynamic(nil, true)
        var sawBall = false
        advance(backend, seconds: 0.5) {
            let read = self.readTransforms(backend)
            XCTAssertNil(read[self.zombie])
            sawBall = sawBall || read[7] != nil
        }
        XCTAssertTrue(sawBall, "the ball, an engine body, is read back as usual")
    }

    func testImpulseMovesADynamicPart() throws {
        let backend = makeBackend()
        // One free part (no joint pulls on it, and Jolt's mass balancing
        // leaves a lone part alone): the impulse's whole momentum is its.
        var single = chain()
        single.parts.removeLast(2)
        single.startActive = true
        let lone = try XCTUnwrap(backend.addRagdoll(single))
        lone.setPartDynamic(0, true)
        lone.addImpulse(SIMD3<Float>(2, 0, 0), toPart: 0, at: SIMD3<Float>(0, 1.15, 0))
        backend.step(deltaTime: step)
        // 2 N s on 10 kg is 0.2 m/s: one step's travel, less Jolt's 5 %/s
        // linear damping.
        XCTAssertEqual(positions(lone)[0].x, 0.2 * step, accuracy: 0.0002)
        XCTAssertEqual(positions(lone)[0].z, 0, accuracy: 1e-5)

        // In the chain, a hit on the one dynamic part swings it about its
        // pivot (which its joint pins, so the shape's centre is what moves);
        // the same hit on a kinematic part does nothing.
        let ragdoll = makeChain(backend)
        ragdoll.setPartDynamic(2, true)
        var pose: [simd_float4x4] = []
        func centreX(_ part: Int) -> Float {
            ragdoll.readPose(into: &pose)
            return pose[part].columns.3.x + 0.15 * pose[part].columns.1.x
        }
        ragdoll.addImpulse(SIMD3<Float>(1, 0, 0), toPart: 1, at: SIMD3<Float>(0, 1.45, 0))
        advance(backend, seconds: 0.1)
        XCTAssertEqual(centreX(1), 0, accuracy: 1e-5, "a kinematic part ignores the impulse")
        ragdoll.addImpulse(SIMD3<Float>(1, 0, 0), toPart: 2, at: SIMD3<Float>(0, 1.75, 0))
        advance(backend, seconds: 0.1)
        XCTAssertEqual(positions(ragdoll)[2].x, 0, accuracy: 1e-3, "the pivot stays on its kinematic parent")
        XCTAssertGreaterThan(centreX(2), 0.01, "the dynamic part swung with the hit")
    }

    func testRagdollsDoNotOutliveTheWorld() {
        // The world destructor must take the ragdoll out and destroy its
        // parts before sweeping the bodies; a crash here is the failure.
        let ragdoll: JoltRagdoll = {
            let backend = makeBackend()
            let ragdoll = makeChain(backend)
            ragdoll.setPartDynamic(nil, true)
            advance(backend, seconds: 0.2)
            return ragdoll // the backend, and its world, die here
        }()
        XCTAssertTrue(ragdoll.isValid, "the handle is not told; the owner must not use it")
    }

    // MARK: - Records, guards, friction

    func testABallOnAKinematicPartIsReportedWithTheBallAsA() {
        let backend = makeBackend()
        let ragdoll = makeChain(backend)
        ragdoll.setKinematicPose(neutralPose())
        // The parts went dynamic and back: the records must have followed,
        // or the ball and the part are both "dynamic" and A is whichever
        // Jolt lists first.
        ragdoll.setPartDynamic(nil, true)
        ragdoll.setPartDynamic(nil, false)
        // A ball dropped onto the lower part's capsule (its top is at
        // 1.6 + 0.3 + 0.05 = 1.95 m).
        backend.didAddBody(entity: 9, descriptor: PhysicsBodyDescriptor(
            motionType: .dynamic,
            collider: PhysicsColliderDescriptor(shape: .sphere(radius: 0.05), friction: 0.5, restitution: 0.0),
            mass: 0.5,
            position: SIMD3<Float>(0, 2.3, 0)
        ))
        let sink = RecordingSink()
        advance(backend, seconds: 1, sink: sink)
        let hits = sink.contacts.filter { $0.phase == .began && ($0.entityA == zombie || $0.entityB == zombie) }
        XCTAssertFalse(hits.isEmpty, "the ball lands on the part")
        for hit in hits {
            XCTAssertEqual(hit.entityA, 9, "the dynamic body is A")
            XCTAssertEqual(hit.entityB, zombie, "the kinematic part is B")
        }
    }

    func testAFarKinematicPoseTeleportsWithoutSpeed() {
        var settings = JoltWorldSettings()
        settings.workerThreads = 0
        settings.maxKinematicStep = 1
        settings.maxKinematicSpeed = 6
        let backend = JoltPhysicsBackend(settings: settings)
        backend.configure(PhysicsWorldConfiguration())
        let ragdoll = makeChain(backend)
        // Far: a jump, with no implied velocity.
        let far = heights.map { transform(SIMD3<Float>(5, $0, 0)) }
        ragdoll.setKinematicPose(far)
        backend.step(deltaTime: step)
        let placed = positions(ragdoll)
        XCTAssertEqual(placed[0].x, 5, accuracy: 1e-3)
        XCTAssertEqual(placed[2].x, 5, accuracy: 1e-3)
        // Near but fast: the parts approach at the cap, not in one step.
        let near = heights.map { transform(SIMD3<Float>(5.5, $0, 0)) }
        ragdoll.setKinematicPose(near)
        backend.step(deltaTime: step)
        let moved = positions(ragdoll)
        XCTAssertLessThan(moved[0].x - placed[0].x, 6 * step + 1e-3, "no faster than the cap")
        XCTAssertGreaterThan(moved[0].x, placed[0].x, "but on its way")
    }

    func testPartFrictionStopsASlidingPart() {
        func travel(friction: Float) -> Float {
            let backend = makeBackend()
            var descriptor = chain()
            descriptor.startActive = true
            for index in descriptor.parts.indices { descriptor.parts[index].friction = friction }
            let ragdoll = backend.addRagdoll(descriptor)!
            // The chain laid flat on the floor, sliding along +X.
            let flat = simd_quatf(angle: -.pi / 2, axis: SIMD3<Float>(0, 0, 1))
            let pose = heights.map { transform(SIMD3<Float>($0 - 1.0, 0.1, 0), flat) }
            ragdoll.setPose(pose, resetVelocities: true)
            ragdoll.setPartDynamic(nil, true)
            ragdoll.setMotors(nil, mode: .off)
            ragdoll.setVelocities(linear: Array(repeating: SIMD3<Float>(3, 0, 0), count: 3), angular: nil)
            for _ in 0 ..< 90 { backend.step(deltaTime: step) }
            return positions(ragdoll)[0].x
        }
        let slippery = travel(friction: 0.02)
        let grippy = travel(friction: 1.0)
        XCTAssertGreaterThan(slippery, grippy + 0.3, "friction shortens the slide: \(slippery) vs \(grippy)")
    }

    /// A two-part hinge hanging from a kinematic root: the child's cone is
    /// centred 60° into flexion (about +Z), so it may fold to 120° that way
    /// and not at all the other.
    func testParentAxesCentreTheConeOffTheNeutralPose() {
        func settle(pushing direction: Float) -> Float {
            let backend = makeBackend()
            var parts: [JoltRagdollPart] = []
            var root = JoltRagdollPart(
                name: "root", parentIndex: nil, shape: .capsule(radius: 0.05, height: 0.2), mass: 5,
                position: SIMD3<Float>(0, 2.0, 0), rotation: identity, twistAxis: SIMD3<Float>(0, 1, 0), planeAxis: SIMD3<Float>(0, 0, 1)
            )
            root.shapeOffset = SIMD3<Float>(0, 0.15, 0)
            parts.append(root)
            // The child hangs down from the root's pivot: its bone is -Y,
            // the flexion axis +Z (a swing of +θ about it carries the bone
            // toward +Z × -Y = +X, so the cone is centred toward +X).
            var child = JoltRagdollPart(
                name: "child", parentIndex: 0, shape: .capsule(radius: 0.04, height: 0.3), mass: 4,
                position: SIMD3<Float>(0, 2.0, 0), rotation: identity, twistAxis: SIMD3<Float>(0, -1, 0), planeAxis: SIMD3<Float>(0, 0, 1)
            )
            child.shapeOffset = SIMD3<Float>(0, -0.19, 0)
            // The bend tilts the bone toward the normal axis (+Z × -Y = +X),
            // which the normal half cone limits; across it, a sliver.
            child.normalHalfConeAngle = 60 * .pi / 180
            child.planeHalfConeAngle = 2 * .pi / 180
            child.twistRange = (-2 * Float.pi / 180) ... (2 * Float.pi / 180)
            let centre = simd_quatf(angle: 60 * .pi / 180, axis: SIMD3<Float>(0, 0, 1))
            child.parentTwistAxis = centre.act(child.twistAxis)
            child.parentPlaneAxis = centre.act(child.planeAxis)
            parts.append(child)
            var descriptor = JoltRagdollDescriptor(parts: parts)
            descriptor.entity = zombie
            descriptor.startActive = true
            let ragdoll = backend.addRagdoll(descriptor)!
            // In constraint space the child starts 60° from the centre.
            let rest = ragdoll.jointRotation(ofPart: 1)!
            XCTAssertEqual(2 * asin(abs(rest.imag.z)) * 180 / .pi, 60, accuracy: 0.5)
            XCTAssertNil(ragdoll.jointRotation(ofPart: 0), "the root has no joint")
            ragdoll.setPartDynamic(1, true)
            ragdoll.setMotors(nil, mode: .off)
            // A steady sideways shove on the child's tip.
            for _ in 0 ..< 120 {
                ragdoll.addImpulse(SIMD3<Float>(direction * 0.6, 0, 0), toPart: 1, at: SIMD3<Float>(0, 1.65, 0))
                backend.step(deltaTime: step)
            }
            var pose: [simd_float4x4] = []
            ragdoll.readPose(into: &pose)
            let bone = pose[1] * SIMD4<Float>(0, -1, 0, 0)
            // The bone's angle from straight down, signed toward +X.
            return atan2(bone.x, -bone.y) * 180 / .pi
        }
        let flexed = settle(pushing: 1)
        let extended = settle(pushing: -1)
        XCTAssertGreaterThan(flexed, 100, "folds well past 60° into flexion: \(flexed)°")
        XCTAssertLessThan(extended, 6, "and not beyond straight the other way: \(extended)°")
        XCTAssertGreaterThan(extended, -6, "nor anywhere near the free pendulum's 60°: \(extended)°")
    }

    func testSleepingCanBeForbiddenWhileABodySettles() {
        func awakeAfterRest(allowSleeping: Bool) -> Bool {
            let backend = makeBackend()
            let ragdoll = makeChain(backend)
            // The chain laid flat on the floor, at rest.
            let flat = simd_quatf(angle: -.pi / 2, axis: SIMD3<Float>(0, 0, 1))
            ragdoll.setPose(heights.map { transform(SIMD3<Float>($0 - 1.0, 0.1, 0), flat) }, resetVelocities: true)
            ragdoll.setPartDynamic(nil, true)
            ragdoll.setMotors(nil, mode: .off)
            ragdoll.setAllowSleeping(allowSleeping)
            advance(backend, seconds: 3)
            return ragdoll.isPartAwake(0)
        }
        XCTAssertFalse(awakeAfterRest(allowSleeping: true), "Jolt sleeps a body at rest")
        XCTAssertTrue(awakeAfterRest(allowSleeping: false), "unless told not to")
    }

    func testDisabledPairsLetPartsPassThroughEachOther() {
        // The lower part folded back onto the pelvis: with the pair
        // disabled the capsules overlap freely; otherwise they push apart.
        func overlapAfterFold(disable: Bool) -> Float {
            let backend = makeBackend()
            var descriptor = chain()
            descriptor.startActive = true
            for index in descriptor.parts.indices {
                descriptor.parts[index].normalHalfConeAngle = .pi
                descriptor.parts[index].planeHalfConeAngle = .pi
            }
            if disable { descriptor.disabledCollisionPairs = [(0, 2)] }
            let ragdoll = backend.addRagdoll(descriptor)!
            // Pelvis fixed (kinematic); upper folded down -Y so the lower
            // part's capsule lands inside the pelvis capsule.
            let down = simd_quatf(angle: .pi, axis: SIMD3<Float>(0, 0, 1))
            ragdoll.setKinematicPose([transform(SIMD3<Float>(0, 1.0, 0)), transform(SIMD3<Float>(0, 1.3, 0), down), transform(SIMD3<Float>(0, 1.0, 0), down)])
            ragdoll.setPartDynamic(2, true)
            ragdoll.setMotors(nil, mode: .off)
            advance(backend, seconds: 1)
            let pelvis = positions(ragdoll)[0], lower = positions(ragdoll)[2]
            return simd_distance(pelvis, lower)
        }
        let apart = overlapAfterFold(disable: false)
        let through = overlapAfterFold(disable: true)
        XCTAssertGreaterThan(apart, through + 0.03, "collision pushes the folded part out (\(apart) vs \(through))")
    }

    func testGravityCanBeTurnedOffPerPartAndAnImpulseAtTheCentrePushesWithoutSpin() {
        let backend = makeBackend()
        var descriptor = chain()
        descriptor.startActive = true
        let ragdoll = backend.addRagdoll(descriptor)!
        // The lower part free of gravity, hanging from a kinematic upper.
        ragdoll.setKinematicPose(neutralPose())
        ragdoll.setPartDynamic(2, true)
        ragdoll.setMotors(nil, mode: .off)
        ragdoll.setGravityFactor(2, 0)
        // The part's origin is its pivot, which the joint holds: watch the
        // far end of the capsule, 0.3 m up its axis, which stands above the
        // pivot and would topple under gravity.
        func tip() -> SIMD3<Float> {
            var pose: [simd_float4x4] = []
            ragdoll.readPose(into: &pose)
            let t = pose[2] * SIMD4<Float>(0, 0.3, 0, 1)
            return SIMD3<Float>(t.x, t.y, t.z)
        }
        let before = tip()
        advance(backend, seconds: 1)
        let held = tip()
        XCTAssertEqual(held.y, before.y, accuracy: 0.02, "no gravity, no topple")

        // A sideways impulse at the centre of mass: the part swings that way.
        ragdoll.addImpulse(SIMD3<Float>(2, 0, 0), toPart: 2)
        advance(backend, seconds: 0.2)
        let pushed = tip()
        XCTAssertGreaterThan(pushed.x, held.x + 0.03, "the push carries it along +X")

        // Gravity back: it topples.
        ragdoll.setGravityFactor(nil, 1)
        advance(backend, seconds: 1)
        XCTAssertLessThan(tip().y, held.y - 0.05, "and gravity takes it down again")
    }

    func testAddedVelocityShovesTheDynamicPartsAndLeavesKinematicOnesAlone() {
        let backend = makeBackend()
        let ragdoll = makeChain(backend)
        ragdoll.setKinematicPose(neutralPose())
        ragdoll.setPartDynamic(2, true)
        ragdoll.setMotors(nil, mode: .off)
        ragdoll.setGravityFactor(nil, 0)
        ragdoll.addLinearVelocity(SIMD3<Float>(0, 0, 3))
        advance(backend, seconds: 0.1)
        var pose: [simd_float4x4] = []
        ragdoll.readPose(into: &pose)
        let tip = pose[2] * SIMD4<Float>(0, 0.3, 0, 1)
        XCTAssertGreaterThan(tip.z, 0.05, "the dynamic part swings along +Z: \(tip.z)")
        XCTAssertEqual(pose[1].columns.3.z, 0, accuracy: 1e-3, "the kinematic part stays where its pose puts it")
    }

    /// How far a powered chain trails a target that swings like an arm at a
    /// run: the lower part's target rotates ±45° about Z at 1.5 Hz. Printed
    /// for tuning; asserted loosely, and sub-steps must help.
    func testTrackingLagBehindASwingingTarget() {
        func lag(frequency: Float, torque: Float, substeps: Int32) -> Float {
            var settings = JoltWorldSettings()
            settings.workerThreads = 0
            settings.collisionSteps = substeps
            let backend = JoltPhysicsBackend(settings: settings)
            backend.configure(PhysicsWorldConfiguration())
            var descriptor = chain()
            descriptor.startActive = true
            let ragdoll = backend.addRagdoll(descriptor)!
            ragdoll.setPartDynamic(1, true)
            ragdoll.setPartDynamic(2, true)
            ragdoll.setMotors(nil, mode: .position, frequency: frequency, damping: 1, maxTorque: torque)
            var worst: Float = 0
            var t: Float = 0
            for frame in 0 ..< 180 {
                t += step
                let angle = 45 * Float.pi / 180 * sin(2 * .pi * 1.5 * t)
                let q = simd_quatf(angle: angle, axis: SIMD3<Float>(0, 0, 1))
                let upperPivot = SIMD3<Float>(0, 1.3, 0)
                let lowerPivot = upperPivot + q.act(SIMD3<Float>(0, 0.3, 0))
                let target = [transform(SIMD3<Float>(0, 1.0, 0)), transform(upperPivot, q), transform(lowerPivot, q)]
                ragdoll.driveMotors(toward: target)
                backend.step(deltaTime: step)
                if frame > 60 {
                    var pose: [simd_float4x4] = []
                    ragdoll.readPose(into: &pose)
                    let tip = pose[2] * SIMD4<Float>(0, 0.3, 0, 1)
                    let wantTip = target[2] * SIMD4<Float>(0, 0.3, 0, 1)
                    worst = max(worst, simd_distance(SIMD3<Float>(tip.x, tip.y, tip.z), SIMD3<Float>(wantTip.x, wantTip.y, wantTip.z)))
                }
            }
            return worst
        }
        for (frequency, torque, substeps) in [(Float(20), Float(500), Int32(1)), (30, 2000, 1), (60, 2000, 3), (90, 2000, 4), (60, 500, 3)] {
            print(String(format: "tracking lag: %.0f Hz, %.0f N m, %d substeps -> tip error %.3f m", frequency, torque, substeps, lag(frequency: frequency, torque: torque, substeps: substeps)))
        }
        XCTAssertLessThan(lag(frequency: 20, torque: 2000, substeps: 1), 0.3)
        XCTAssertLessThan(lag(frequency: 60, torque: 2000, substeps: 3), lag(frequency: 20, torque: 2000, substeps: 1) / 2, "sub-steps let the motors run stiffer and halve the lag")
    }
}
