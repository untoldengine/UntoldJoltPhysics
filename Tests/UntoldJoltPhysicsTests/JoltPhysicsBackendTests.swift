//
//  JoltPhysicsBackendTests.swift
//  UntoldJoltPhysicsTests
//
//  Drives the backend through the engine's PhysicsBackend protocol exactly
//  as the PhysicsCoordinator would — the same scenarios the pure-Swift
//  CoolBasket backend is tested with, so the two backends stay interchangeable.
//

import simd
@testable import UntoldJoltPhysics
import UntoldEngine
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

final class JoltPhysicsBackendTests: XCTestCase {
    private let step: Float = 1.0 / 60.0

    private func makeBackend(floor: Bool = true) -> JoltPhysicsBackend {
        var settings = JoltWorldSettings()
        settings.workerThreads = 0 // deterministic single-threaded stepping in tests
        let backend = JoltPhysicsBackend(settings: settings)
        backend.configure(PhysicsWorldConfiguration())
        if floor {
            // Ground: a big static slab whose top face is y = 0.
            backend.didAddBody(entity: 1000, descriptor: PhysicsBodyDescriptor(
                motionType: .static,
                collider: PhysicsColliderDescriptor(
                    shape: .box(halfExtents: simd_float3(50, 0.5, 50)),
                    friction: 0.5,
                    restitution: 0.0
                ),
                position: simd_float3(0, -0.5, 0)
            ))
        }
        return backend
    }

    private func ball(
        position: simd_float3,
        velocity: simd_float3 = .zero,
        restitution: Float = 0.6,
        radius: Float = 0.11
    ) -> PhysicsBodyDescriptor {
        PhysicsBodyDescriptor(
            motionType: .dynamic,
            collider: PhysicsColliderDescriptor(
                shape: .sphere(radius: radius),
                friction: 0.35,
                restitution: restitution
            ),
            mass: 0.43,
            position: position,
            linearVelocity: velocity
        )
    }

    private func advance(_ backend: JoltPhysicsBackend, seconds: Float, sink: RecordingSink? = nil) {
        var elapsed: Float = 0
        while elapsed < seconds {
            backend.step(deltaTime: step)
            if let sink { backend.drainEvents(into: sink) }
            elapsed += step
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

    private func writeKinematic(_ backend: JoltPhysicsBackend, entity: EntityID, position: simd_float3) {
        var entities: [EntityID] = [entity]
        var transforms = [PhysicsBodyTransform(position: position, orientation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1))]
        entities.withUnsafeBufferPointer { entityBuffer in
            transforms.withUnsafeBufferPointer { transformBuffer in
                backend.writeKinematicTargets(PhysicsBodyWriteBatch(entities: entityBuffer, transforms: transformBuffer))
            }
        }
    }

    // MARK: - Tests

    func testCapabilitiesAndIdentity() {
        let backend = makeBackend(floor: false)
        XCTAssertEqual(backend.id, JoltPhysicsPluginContract.backendID)
        XCTAssertTrue(backend.capabilities.contains(.collisions))
        XCTAssertTrue(backend.capabilities.contains(.triggers))
        XCTAssertTrue(backend.capabilities.contains(.raycast))
        XCTAssertEqual(backend.bodyCount, 0)
    }

    func testBallFallsBouncesAndIsReadBack() {
        let backend = makeBackend()
        backend.didAddBody(entity: 1, descriptor: ball(position: simd_float3(0, 1.0, 0)))
        let sink = RecordingSink()

        advance(backend, seconds: 0.6, sink: sink)
        let state = backend.bodyState(for: 1)!
        XCTAssertGreaterThanOrEqual(state.position.y, 0.10, "Ball must not sink through the floor")
        XCTAssertLessThan(state.position.y, 1.0, "Rebound cannot exceed the drop height")

        let transforms = readTransforms(backend)
        XCTAssertNotNil(transforms[1], "An active dynamic body must be read back")
        XCTAssertNil(transforms[1000], "Static bodies are never read back")
        XCTAssertEqual(transforms[1]!.position.y, state.position.y, accuracy: 1e-5)

        let floorContacts = sink.contacts.filter { $0.phase == .began && ($0.entityA == 1000 || $0.entityB == 1000) }
        XCTAssertFalse(floorContacts.isEmpty, "Hitting the floor must report a contact")
        XCTAssertGreaterThan(floorContacts.first!.impulse, 0.5, "A 1 m drop of a 0.43 kg ball is a firm impulse")
        XCTAssertEqual(floorContacts.first!.entityA, 1, "A is the dynamic body")
        XCTAssertEqual(floorContacts.first!.entityB, 1000)
        XCTAssertEqual(floorContacts.first!.normal.y, 1.0, accuracy: 1e-3, "Normal points out of the floor into the ball")
    }

    func testBallEventuallyRestsAndFallsAsleep() {
        let backend = makeBackend()
        backend.didAddBody(entity: 1, descriptor: ball(position: simd_float3(0, 0.8, 0)))
        let sink = RecordingSink()

        advance(backend, seconds: 6.0, sink: sink)
        let resting = backend.bodyState(for: 1)!
        XCTAssertEqual(resting.position.y, 0.11, accuracy: 0.02)
        XCTAssertLessThan(simd_length(resting.velocity), 0.05)
        XCTAssertFalse(backend.isBodyActive(entity: 1), "Jolt puts a resting ball to sleep")
        XCTAssertTrue(sink.activations.contains { $0.entity == 1 && !$0.isActive }, "Falling asleep is reported")
        XCTAssertTrue(readTransforms(backend).isEmpty, "A sleeping body is not read back every step")
    }

    func testBallBouncesOffStaticBox() {
        let backend = makeBackend()
        backend.didAddBody(entity: 10, descriptor: PhysicsBodyDescriptor(
            motionType: .static,
            collider: PhysicsColliderDescriptor(
                shape: .box(halfExtents: simd_float3(0.05, 1.0, 1.0)),
                restitution: 0.7
            ),
            position: simd_float3(1.0, 0.11, 0)
        ))
        backend.didAddBody(entity: 1, descriptor: ball(
            position: simd_float3(0, 0.11, 0),
            velocity: simd_float3(2.0, 0, 0)
        ))
        let sink = RecordingSink()

        advance(backend, seconds: 1.0, sink: sink)
        let state = backend.bodyState(for: 1)!
        XCTAssertLessThan(state.velocity.x, 0, "Ball must rebound off the box")
        XCTAssertLessThan(state.position.x, 0.9)
        let wallPhases = sink.contacts.filter { $0.entityB == 10 }.map(\.phase)
        XCTAssertEqual(wallPhases.first, .began)
        XCTAssertEqual(wallPhases.last, .ended, "The bounce's contact ends once the ball leaves")
        XCTAssertTrue(sink.contacts.filter { $0.entityB == 10 }.allSatisfy { $0.entityA == 1 }, "A is the ball")
    }

    func testRaysPassThroughTriggers() {
        let backend = makeBackend(floor: false)
        backend.didAddBody(entity: 20, descriptor: PhysicsBodyDescriptor(
            motionType: .static,
            collider: PhysicsColliderDescriptor(shape: .box(halfExtents: simd_float3(0.5, 0.5, 0.5)), isTrigger: true),
            position: simd_float3(2.0, 0, 0)
        ))
        backend.didAddBody(entity: 1, descriptor: PhysicsBodyDescriptor(
            motionType: .static,
            collider: PhysicsColliderDescriptor(shape: .box(halfExtents: simd_float3(0.5, 0.5, 0.5))),
            position: simd_float3(4.0, 0, 0)
        ))
        let hit = backend.raycast(PhysicsRay(origin: .zero, direction: simd_float3(1, 0, 0), maxDistance: 10), filter: PhysicsQueryFilter())
        XCTAssertEqual(hit?.entity, 1, "The trigger in front is not solid")
        XCTAssertEqual(hit?.distance ?? 0, 3.5, accuracy: 1e-3)
    }

    func testKinematicBodiesDoNotTriggerSensors() {
        let backend = makeBackend()
        backend.didAddBody(entity: 20, descriptor: PhysicsBodyDescriptor(
            motionType: .static,
            collider: PhysicsColliderDescriptor(shape: .box(halfExtents: simd_float3(0.3, 0.3, 0.3)), isTrigger: true),
            position: simd_float3(1.0, 0.3, 0)
        ))
        backend.didAddBody(entity: 30, descriptor: PhysicsBodyDescriptor(
            motionType: .kinematic,
            collider: PhysicsColliderDescriptor(shape: .sphere(radius: 0.07)),
            position: simd_float3(-0.5, 0.3, 0)
        ))
        let sink = RecordingSink()
        var handX: Float = -0.5
        for _ in 0 ..< 60 {
            handX += 2.5 * step
            writeKinematic(backend, entity: 30, position: simd_float3(handX, 0.3, 0))
            backend.step(deltaTime: step)
            backend.drainEvents(into: sink)
        }
        XCTAssertGreaterThan(handX, 1.5, "The hand swept through the trigger")
        XCTAssertTrue(sink.triggers.isEmpty, "A hand proxy is not a trigger occupant")
    }

    func testTriggerFiresEnterAndExit() {
        let backend = makeBackend()
        backend.didAddBody(entity: 20, descriptor: PhysicsBodyDescriptor(
            motionType: .static,
            collider: PhysicsColliderDescriptor(
                shape: .box(halfExtents: simd_float3(0.2, 0.5, 0.5)),
                isTrigger: true
            ),
            position: simd_float3(1.0, 0.11, 0)
        ))
        backend.didAddBody(entity: 1, descriptor: ball(
            position: simd_float3(0, 0.11, 0),
            velocity: simd_float3(3.0, 0, 0)
        ))
        let sink = RecordingSink()

        advance(backend, seconds: 1.5, sink: sink)
        let phases = sink.triggers
            .filter { $0.triggerEntity == 20 && $0.otherEntity == 1 }
            .map(\.phase)
        XCTAssertEqual(phases, [.entered, .exited], "Ball passing through must enter then exit")
        XCTAssertFalse(sink.contacts.contains { $0.entityA == 20 || $0.entityB == 20 }, "Sensors never surface as contacts")
        XCTAssertGreaterThan(backend.bodyState(for: 1)!.position.x, 1.3, "A trigger must not stop the ball")
    }

    func testRemovingABodyInsideATriggerStillReportsTheExit() {
        let backend = makeBackend()
        backend.didAddBody(entity: 20, descriptor: PhysicsBodyDescriptor(
            motionType: .static,
            collider: PhysicsColliderDescriptor(
                shape: .box(halfExtents: simd_float3(0.5, 0.5, 0.5)),
                isTrigger: true
            ),
            position: simd_float3(0, 0.5, 0)
        ))
        backend.didAddBody(entity: 1, descriptor: ball(position: simd_float3(0, 0.11, 0)))
        let sink = RecordingSink()
        advance(backend, seconds: 0.2, sink: sink)
        XCTAssertEqual(sink.triggers.map(\.phase), [.entered])

        // The ball is grabbed (its body removed) while inside the trigger.
        backend.didRemoveBody(entity: 1)
        advance(backend, seconds: 0.1, sink: sink)
        XCTAssertEqual(sink.triggers.map(\.phase), [.entered, .exited], "The exit must still name the removed body")
        XCTAssertEqual(sink.triggers.last?.otherEntity, 1)
    }

    func testSleepingOccupantStaysInsideATrigger() {
        let backend = makeBackend()
        backend.didAddBody(entity: 20, descriptor: PhysicsBodyDescriptor(
            motionType: .static,
            collider: PhysicsColliderDescriptor(
                shape: .box(halfExtents: simd_float3(0.5, 0.5, 0.5)),
                isTrigger: true
            ),
            position: simd_float3(0, 0.5, 0)
        ))
        backend.didAddBody(entity: 1, descriptor: ball(position: simd_float3(0, 0.3, 0)))
        let sink = RecordingSink()
        advance(backend, seconds: 6.0, sink: sink)
        XCTAssertFalse(backend.isBodyActive(entity: 1), "The ball fell asleep inside the trigger")
        XCTAssertEqual(sink.triggers.map(\.phase), [.entered], "Sleeping inside is not an exit")
        XCTAssertFalse(sink.contacts.contains { $0.entityA == 20 || $0.entityB == 20 })
        XCTAssertNil(readTransforms(backend)[20], "A trigger is never read back")
    }

    func testMovingKinematicHandSwatsTheBall() {
        let backend = makeBackend()
        backend.didAddBody(entity: 1, descriptor: ball(position: simd_float3(0, 0.11, 0)))
        backend.didAddBody(entity: 30, descriptor: PhysicsBodyDescriptor(
            motionType: .kinematic,
            collider: PhysicsColliderDescriptor(shape: .sphere(radius: 0.07)),
            position: simd_float3(-0.5, 0.11, 0)
        ))
        let sink = RecordingSink()

        // Sweep the hand into the ball at 2.5 m/s using kinematic targets.
        var handX: Float = -0.5
        for _ in 0 ..< 30 {
            handX += 2.5 * step
            writeKinematic(backend, entity: 30, position: simd_float3(handX, 0.11, 0))
            backend.step(deltaTime: step)
            backend.drainEvents(into: sink)
        }

        let state = backend.bodyState(for: 1)!
        XCTAssertGreaterThan(state.velocity.x, 1.0, "The swat must transfer hand momentum")
        XCTAssertTrue(sink.contacts.contains { $0.entityA == 30 || $0.entityB == 30 })
        XCTAssertNil(readTransforms(backend)[30], "Kinematic bodies are driven by the engine, never read back")
    }

    func testRemovingABodyWakesWhatRestedOnIt() {
        let backend = makeBackend()
        // A crate on the floor, a ball asleep on the crate.
        backend.didAddBody(entity: 7, descriptor: PhysicsBodyDescriptor(
            motionType: .dynamic,
            collider: PhysicsColliderDescriptor(shape: .box(halfExtents: simd_float3(0.3, 0.2, 0.3)), friction: 0.5, restitution: 0.0),
            mass: 5,
            position: simd_float3(0, 0.2, 0)
        ))
        backend.didAddBody(entity: 8, descriptor: ball(position: simd_float3(0, 0.4 + 0.11, 0), restitution: 0.0))
        advance(backend, seconds: 3.0)
        XCTAssertFalse(backend.isBodyActive(entity: 8), "The ball fell asleep on the crate")
        let before = backend.bodyState(for: 8)!.position.y

        backend.didRemoveBody(entity: 7)
        advance(backend, seconds: 0.5)
        let after = backend.bodyState(for: 8)!.position.y
        XCTAssertLessThan(after, before - 0.1, "With the crate gone the ball falls instead of hovering asleep")
    }

    func testRemovedBodyStopsSimulating() {
        let backend = makeBackend()
        backend.didAddBody(entity: 1, descriptor: ball(position: simd_float3(0, 1.0, 0)))
        XCTAssertEqual(backend.bodyCount, 2)
        backend.didRemoveBody(entity: 1)
        XCTAssertEqual(backend.bodyCount, 1)
        advance(backend, seconds: 0.2)
        XCTAssertNil(backend.bodyState(for: 1))
        XCTAssertNil(readTransforms(backend)[1])
    }

    func testReAddingAnEntityReplacesItsBody() {
        let backend = makeBackend()
        backend.didAddBody(entity: 1, descriptor: ball(position: simd_float3(0, 1.0, 0)))
        backend.didAddBody(entity: 1, descriptor: ball(position: simd_float3(2.0, 0.11, 0)))
        XCTAssertEqual(backend.bodyCount, 2)
        XCTAssertEqual(backend.bodyState(for: 1)!.position.x, 2.0, accuracy: 1e-5)
    }

    func testResetBodyTeleports() {
        let backend = makeBackend()
        backend.didAddBody(entity: 1, descriptor: ball(position: simd_float3(0, 0.11, 0), velocity: simd_float3(3, 0, 0)))
        advance(backend, seconds: 0.5)
        XCTAssertGreaterThan(backend.bodyState(for: 1)!.position.x, 0.5)
        XCTAssertTrue(backend.resetBody(entity: 1, position: simd_float3(0, 1.0, 0), velocity: .zero))
        let state = backend.bodyState(for: 1)!
        XCTAssertEqual(state.position.y, 1.0, accuracy: 1e-5)
        XCTAssertEqual(simd_length(state.velocity), 0, accuracy: 1e-5)
        XCTAssertFalse(backend.resetBody(entity: 99, position: .zero, velocity: .zero))
    }

    func testRaycastHitsNearestBodyAndHonorsExclusions() {
        let backend = makeBackend(floor: false)
        for (entity, x) in [(EntityID(1), Float(2.0)), (EntityID(2), Float(4.0))] {
            backend.didAddBody(entity: entity, descriptor: PhysicsBodyDescriptor(
                motionType: .static,
                collider: PhysicsColliderDescriptor(shape: .box(halfExtents: simd_float3(0.5, 0.5, 0.5))),
                position: simd_float3(x, 0, 0)
            ))
        }
        let ray = PhysicsRay(origin: .zero, direction: simd_float3(1, 0, 0), maxDistance: 10)

        let nearest = backend.raycast(ray, filter: PhysicsQueryFilter())
        XCTAssertEqual(nearest?.entity, 1)
        XCTAssertEqual(nearest?.distance ?? 0, 1.5, accuracy: 1e-3)
        XCTAssertEqual(nearest?.position.x ?? 0, 1.5, accuracy: 1e-3)
        XCTAssertEqual(nearest?.normal.x ?? 0, -1.0, accuracy: 1e-3)

        let behind = backend.raycast(ray, filter: PhysicsQueryFilter(excludedEntities: [1]))
        XCTAssertEqual(behind?.entity, 2)
        XCTAssertEqual(behind?.distance ?? 0, 3.5, accuracy: 1e-3)

        XCTAssertNil(backend.raycast(PhysicsRay(origin: .zero, direction: simd_float3(0, 1, 0), maxDistance: 10), filter: PhysicsQueryFilter()))
        XCTAssertNil(backend.raycast(PhysicsRay(origin: .zero, direction: simd_float3(1, 0, 0), maxDistance: 1.0), filter: PhysicsQueryFilter()), "Beyond maxDistance is a miss")
    }

    func testRaycastLayerMask() {
        let backend = makeBackend(floor: false)
        backend.didAddBody(entity: 1, descriptor: PhysicsBodyDescriptor(
            motionType: .static,
            collider: PhysicsColliderDescriptor(shape: .box(halfExtents: simd_float3(0.5, 0.5, 0.5))),
            layer: 3,
            position: simd_float3(2.0, 0, 0)
        ))
        let ray = PhysicsRay(origin: .zero, direction: simd_float3(1, 0, 0), maxDistance: 10)
        XCTAssertNotNil(backend.raycast(ray, filter: PhysicsQueryFilter(layerMask: 1 << 3)))
        XCTAssertNil(backend.raycast(ray, filter: PhysicsQueryFilter(layerMask: 1 << 2)))
    }

    func testCollisionLayerMatrixDisablesPairs() {
        // Layer 1 (ball) collides only with layer 0 (floor); layer 2 (wall) is transparent to it.
        let backend = makeBackend()
        var config = PhysicsWorldConfiguration()
        config.collisionLayerMatrix = [0b111, 0b001, 0b001]
        backend.configure(config)
        backend.didAddBody(entity: 10, descriptor: PhysicsBodyDescriptor(
            motionType: .static,
            collider: PhysicsColliderDescriptor(shape: .box(halfExtents: simd_float3(0.05, 1.0, 1.0))),
            layer: 2,
            position: simd_float3(1.0, 0.11, 0)
        ))
        var ballDescriptor = ball(position: simd_float3(0, 0.11, 0), velocity: simd_float3(2.0, 0, 0))
        ballDescriptor.layer = 1
        backend.didAddBody(entity: 1, descriptor: ballDescriptor)

        let sink = RecordingSink()
        advance(backend, seconds: 1.0, sink: sink)
        XCTAssertGreaterThan(backend.bodyState(for: 1)!.position.x, 1.2, "The wall's layer is filtered out, the ball passes through")
        XCTAssertEqual(backend.bodyState(for: 1)!.position.y, 0.11, accuracy: 0.03, "The floor's layer is still solid")
        // (No floor impact is reported: the ball starts resting on it.)
        XCTAssertFalse(sink.contacts.contains { $0.entityB == 10 })
    }

    func testCapsuleCylinderAndConvexHullShapesSimulate() {
        let backend = makeBackend()
        backend.didAddBody(entity: 1, descriptor: PhysicsBodyDescriptor(
            motionType: .dynamic,
            collider: PhysicsColliderDescriptor(shape: .capsule(radius: 0.1, height: 0.4)),
            position: simd_float3(-1, 1.0, 0)
        ))
        backend.didAddBody(entity: 2, descriptor: PhysicsBodyDescriptor(
            motionType: .dynamic,
            collider: PhysicsColliderDescriptor(shape: .cylinder(radius: 0.1, height: 0.4)),
            position: simd_float3(0, 1.0, 0)
        ))
        // A 0.3 m cube centred on the origin, given as its eight corners.
        var cube: [simd_float3] = []
        for x: Float in [-0.15, 0.15] {
            for y: Float in [-0.15, 0.15] {
                for z: Float in [-0.15, 0.15] {
                    cube.append(simd_float3(x, y, z))
                }
            }
        }
        backend.didAddBody(entity: 3, descriptor: PhysicsBodyDescriptor(
            motionType: .dynamic,
            collider: PhysicsColliderDescriptor(shape: .convexHull(vertices: cube)),
            position: simd_float3(1, 1.0, 0)
        ))
        XCTAssertEqual(backend.bodyCount, 4)

        advance(backend, seconds: 3.0)
        for entity: EntityID in [1, 2, 3] {
            let y = backend.bodyState(for: entity)!.position.y
            XCTAssertGreaterThan(y, 0.05, "Body \(entity) must rest on the floor, not fall through")
            XCTAssertLessThan(y, 0.5, "Body \(entity) must have come down")
        }
    }

    func testLocalOffsetShiftsTheCollider() {
        // A sphere whose collider hangs 0.5 m below the body origin rests with
        // its origin 0.5 m above the surface contact.
        let backend = makeBackend()
        backend.didAddBody(entity: 1, descriptor: PhysicsBodyDescriptor(
            motionType: .dynamic,
            collider: PhysicsColliderDescriptor(
                shape: .sphere(radius: 0.1),
                localOffset: simd_float3(0, -0.5, 0)
            ),
            position: simd_float3(0, 1.0, 0)
        ))
        advance(backend, seconds: 3.0)
        XCTAssertEqual(backend.bodyState(for: 1)!.position.y, 0.6, accuracy: 0.03)
    }

    func testEnvironmentBoxesCollideAndReportEntityZero() {
        let backend = makeBackend(floor: false)
        // A tilted table top, 1 m up, as a detected surface.
        backend.setEnvironmentBoxes([
            JoltEnvironmentBox(
                center: simd_float3(0, 1.0 - 0.02, 0),
                halfExtents: simd_float3(0.5, 0.02, 0.5),
                friction: 0.5,
                restitution: 0.0
            ),
        ])
        XCTAssertEqual(backend.environmentBodyCount, 0, "Queued until the next step")
        backend.didAddBody(entity: 1, descriptor: ball(position: simd_float3(0, 1.5, 0)))
        let sink = RecordingSink()
        advance(backend, seconds: 3.0, sink: sink)
        XCTAssertEqual(backend.environmentBodyCount, 1)
        XCTAssertEqual(backend.bodyState(for: 1)!.position.y, 1.11, accuracy: 0.03, "Ball rests on the table")
        let tableContact = sink.contacts.first { $0.phase == .began }
        XCTAssertNotNil(tableContact)
        XCTAssertEqual(tableContact?.entityA, 1)
        XCTAssertEqual(tableContact?.entityB, JoltPhysicsBackend.environmentEntity, "Environment contacts name the null entity")

        // Re-sending the same box keeps its body: no new impact is reported
        // and the resting ball does not even wake.
        let table = JoltEnvironmentBox(
            center: simd_float3(0, 1.0 - 0.02, 0),
            halfExtents: simd_float3(0.5, 0.02, 0.5),
            friction: 0.5,
            restitution: 0.0
        )
        let began = sink.contacts.filter { $0.phase == .began }.count
        backend.setEnvironmentBoxes([table])
        advance(backend, seconds: 0.5, sink: sink)
        XCTAssertEqual(sink.contacts.filter { $0.phase == .began }.count, began, "Unchanged geometry re-sent: no new impact")
        XCTAssertFalse(backend.isBodyActive(entity: 1), "…and the ball stayed asleep")

        // A changed box is rebuilt; the re-established resting contact is
        // still not an impact.
        var wider = table
        wider.halfExtents.x = 0.6
        backend.setEnvironmentBoxes([wider])
        advance(backend, seconds: 0.5, sink: sink)
        XCTAssertEqual(sink.contacts.filter { $0.phase == .began }.count, began, "Resting contact re-established: no impact")
        XCTAssertEqual(backend.bodyState(for: 1)!.position.y, 1.11, accuracy: 0.03)

        // Replacing the environment drops the old surface: the ball falls,
        // and the rebuilt contact, never reported as began, does not end.
        let ended = sink.contacts.filter { $0.phase == .ended }.count
        backend.setEnvironmentBoxes([])
        advance(backend, seconds: 1.0, sink: sink)
        XCTAssertEqual(backend.environmentBodyCount, 0)
        XCTAssertLessThan(backend.bodyState(for: 1)!.position.y, 0.5)
        XCTAssertEqual(sink.contacts.filter { $0.phase == .ended }.count, ended, "No ended without a reported began")
    }

    func testEnvironmentBoxLayerIsFilteredByTheMatrix() {
        // A shelf 1 m up on layer 2; the ball is on layer 1, which the matrix
        // lets meet the floor (layer 0) only.
        let backend = makeBackend()
        var config = PhysicsWorldConfiguration()
        config.collisionLayerMatrix = [0b111, 0b001, 0b001]
        backend.configure(config)
        let shelf = JoltEnvironmentBox(
            center: simd_float3(0, 1.0 - 0.02, 0),
            halfExtents: simd_float3(0.5, 0.02, 0.5),
            layer: 2
        )
        XCTAssertEqual(JoltEnvironmentBox(center: .zero, halfExtents: simd_float3(1, 1, 1)).layer, 0, "Boxes are on layer 0 unless said otherwise")
        var relabelled = shelf
        relabelled.layer = 0
        XCTAssertNotEqual(shelf, relabelled, "The layer is part of the box's identity: a change rebuilds it")
        backend.setEnvironmentBoxes([shelf])
        var ballDescriptor = ball(position: simd_float3(0, 1.5, 0), restitution: 0.0)
        ballDescriptor.layer = 1
        backend.didAddBody(entity: 1, descriptor: ballDescriptor)
        advance(backend, seconds: 2.0)
        XCTAssertEqual(backend.environmentBodyCount, 1)
        XCTAssertEqual(backend.bodyState(for: 1)!.position.y, 0.11, accuracy: 0.03, "The shelf's layer is filtered out: the ball falls through it onto the floor")
        backend.didRemoveBody(entity: 1)

        // Let layer 1 meet layer 2: the same shelf, untouched, now holds a ball.
        config.collisionLayerMatrix = [0b111, 0b101, 0b011]
        backend.configure(config)
        backend.didAddBody(entity: 2, descriptor: ballDescriptor)
        advance(backend, seconds: 2.0)
        XCTAssertEqual(backend.environmentBodyCount, 1)
        XCTAssertEqual(backend.bodyState(for: 2)!.position.y, 1.11, accuracy: 0.03, "The matrix change reaches the existing box")
        backend.didRemoveBody(entity: 2)

        // A box that says nothing is on layer 0, the floor's: under the first
        // matrix a layer-1 ball rests on it.
        config.collisionLayerMatrix = [0b111, 0b001, 0b001]
        backend.configure(config)
        var ledge = shelf
        ledge.layer = 0
        ledge.center.x = 2
        backend.setEnvironmentBoxes([shelf, ledge])
        ballDescriptor.position = simd_float3(2, 1.5, 0)
        backend.didAddBody(entity: 3, descriptor: ballDescriptor)
        advance(backend, seconds: 2.0)
        XCTAssertEqual(backend.environmentBodyCount, 2)
        XCTAssertEqual(backend.bodyState(for: 3)!.position.y, 1.11, accuracy: 0.03, "A box on the default layer is as solid as before")
    }

    func testAMatrixChangeWakesWhatTheClosedPairHeldUp() {
        // A layer-1 ball asleep on a layer-2 shelf when the matrix closes the
        // pair: Jolt pairs active bodies only, so the ball would sleep on
        // through the shelf unless the change woke it.
        let backend = makeBackend()
        var config = PhysicsWorldConfiguration()
        config.collisionLayerMatrix = [0b111, 0b101, 0b011]
        backend.configure(config)
        backend.setEnvironmentBoxes([JoltEnvironmentBox(
            center: simd_float3(0, 1.0 - 0.02, 0),
            halfExtents: simd_float3(0.5, 0.02, 0.5),
            layer: 2
        )])
        var ballDescriptor = ball(position: simd_float3(0, 1.2, 0), restitution: 0.0)
        ballDescriptor.layer = 1
        backend.didAddBody(entity: 1, descriptor: ballDescriptor)
        advance(backend, seconds: 2.0)
        XCTAssertEqual(backend.bodyState(for: 1)!.position.y, 1.11, accuracy: 0.03, "Rests on the shelf")
        XCTAssertFalse(backend.isBodyActive(entity: 1), "…asleep")

        config.collisionLayerMatrix = [0b111, 0b001, 0b001]
        backend.configure(config)
        XCTAssertTrue(backend.isBodyActive(entity: 1), "The matrix change woke it")
        advance(backend, seconds: 2.0)
        XCTAssertEqual(backend.bodyState(for: 1)!.position.y, 0.11, accuracy: 0.03, "The closed pair lets the sleeping ball through")

        // The same matrix again is not a change: nothing wakes.
        XCTAssertFalse(backend.isBodyActive(entity: 1), "Asleep on the floor")
        backend.configure(config)
        XCTAssertFalse(backend.isBodyActive(entity: 1), "A repeat of the matrix wakes nothing")
    }

    func testKinematicSpeedCapLimitsWhatAGlitchCanImpart() {
        var settings = JoltWorldSettings()
        settings.workerThreads = 0
        settings.maxKinematicStep = 1.0
        settings.maxKinematicSpeed = 6.0
        let backend = JoltPhysicsBackend(settings: settings)
        backend.configure(PhysicsWorldConfiguration())
        backend.didAddBody(entity: 1000, descriptor: PhysicsBodyDescriptor(
            motionType: .static,
            collider: PhysicsColliderDescriptor(shape: .box(halfExtents: simd_float3(50, 0.5, 50))),
            position: simd_float3(0, -0.5, 0)
        ))
        backend.didAddBody(entity: 1, descriptor: ball(position: simd_float3(0, 0.11, 0)))
        backend.didAddBody(entity: 30, descriptor: PhysicsBodyDescriptor(
            motionType: .kinematic,
            collider: PhysicsColliderDescriptor(shape: .sphere(radius: 0.07)),
            position: simd_float3(-0.19, 0.11, 0)
        ))
        // The hand rests against the ball, then its next target jumps 30 cm
        // through it in one substep (18 m/s implied) — a tracking hiccup.
        writeKinematic(backend, entity: 30, position: simd_float3(-0.18, 0.11, 0))
        backend.step(deltaTime: step)
        writeKinematic(backend, entity: 30, position: simd_float3(0.12, 0.11, 0))
        for _ in 0 ..< 6 {
            backend.step(deltaTime: step)
        }
        let speed = simd_length(backend.bodyState(for: 1)!.velocity)
        XCTAssertLessThan(speed, 12.0, "The ball can receive at most (1 + e) × the cap")
        XCTAssertGreaterThan(speed, 0.5, "…but it was still hit")
    }

    func testFastBallDoesNotTunnelAThinSlab() {
        let backend = makeBackend(floor: false)
        backend.setEnvironmentBoxes([
            JoltEnvironmentBox(center: simd_float3(0, -0.02, 0), halfExtents: simd_float3(5, 0.02, 5)),
        ])
        backend.step(deltaTime: step)
        // 30 m/s straight down at a 4 cm slab: half a metre per substep.
        backend.didAddBody(entity: 1, descriptor: ball(position: simd_float3(0, 1.0, 0), velocity: simd_float3(0, -30, 0)))
        advance(backend, seconds: 0.5)
        XCTAssertGreaterThan(backend.bodyState(for: 1)!.position.y, 0.0, "Continuous collision keeps the ball above the slab")
    }

    func testKinematicJumpBeyondTheStepCapTeleportsWithoutLaunching() {
        var settings = JoltWorldSettings()
        settings.workerThreads = 0
        settings.maxKinematicStep = 1.0
        let backend = JoltPhysicsBackend(settings: settings)
        backend.configure(PhysicsWorldConfiguration())
        backend.didAddBody(entity: 1000, descriptor: PhysicsBodyDescriptor(
            motionType: .static,
            collider: PhysicsColliderDescriptor(shape: .box(halfExtents: simd_float3(50, 0.5, 50))),
            position: simd_float3(0, -0.5, 0)
        ))
        backend.didAddBody(entity: 1, descriptor: ball(position: simd_float3(0, 0.11, 0)))
        // Parked hand far below, then re-appearing right next to the ball.
        backend.didAddBody(entity: 30, descriptor: PhysicsBodyDescriptor(
            motionType: .kinematic,
            collider: PhysicsColliderDescriptor(shape: .sphere(radius: 0.07)),
            position: simd_float3(0, -100, 0)
        ))
        writeKinematic(backend, entity: 30, position: simd_float3(-0.15, 0.11, 0))
        backend.step(deltaTime: step)
        backend.step(deltaTime: step)
        let state = backend.bodyState(for: 1)!
        XCTAssertLessThan(simd_length(state.velocity), 0.5, "A teleporting hand must not swat the ball")

        // A normal sweep still transfers momentum.
        var handX: Float = -0.15
        for _ in 0 ..< 20 {
            handX += 2.5 * step
            writeKinematic(backend, entity: 30, position: simd_float3(handX, 0.11, 0))
            backend.step(deltaTime: step)
        }
        XCTAssertGreaterThan(backend.bodyState(for: 1)!.velocity.x, 1.0)
    }

    func testGravityConfigurationApplies() {
        let backend = makeBackend(floor: false)
        var config = PhysicsWorldConfiguration()
        config.gravity = simd_float3(0, 0, 0)
        backend.configure(config)
        backend.didAddBody(entity: 1, descriptor: ball(position: simd_float3(0, 1.0, 0)))
        advance(backend, seconds: 1.0)
        XCTAssertEqual(backend.bodyState(for: 1)!.position.y, 1.0, accuracy: 1e-4, "Zero gravity: the ball floats")
    }
}
