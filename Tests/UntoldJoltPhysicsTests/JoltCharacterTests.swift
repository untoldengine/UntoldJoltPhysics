//
//  JoltCharacterTests.swift
//  UntoldJoltPhysics
//

import simd
import UntoldEngine
@testable import UntoldJoltPhysics
import XCTest

final class JoltCharacterTests: XCTestCase {
    private let step: Float = 1.0 / 60.0
    private let radius: Float = 0.2
    private let height: Float = 1.8
    private let zombie: EntityID = 42

    /// A world with a static floor whose top is y = 0, and a wall whose face
    /// toward the origin is at x = 0.95 (an environment box, applied by the
    /// first step).
    private func makeBackend(wall: Bool = true) -> JoltPhysicsBackend {
        var settings = JoltWorldSettings()
        settings.workerThreads = 0
        let backend = JoltPhysicsBackend(settings: settings)
        backend.configure(PhysicsWorldConfiguration())
        backend.didAddBody(entity: 1000, descriptor: PhysicsBodyDescriptor(
            motionType: .static,
            collider: PhysicsColliderDescriptor(shape: .box(halfExtents: simd_float3(50, 0.5, 50))),
            position: simd_float3(0, -0.5, 0)
        ))
        if wall {
            backend.setEnvironmentBoxes([
                JoltEnvironmentBox(center: simd_float3(1.0, 0.75, 0), halfExtents: simd_float3(0.05, 0.75, 2)),
            ])
        }
        backend.step(deltaTime: step)
        return backend
    }

    private func makeCharacter(_ backend: JoltPhysicsBackend, at position: simd_float3 = .zero,
                               shape: JoltCharacterDescriptor.Shape = .capsule) -> JoltCharacter
    {
        var descriptor = JoltCharacterDescriptor(radius: radius, height: height)
        descriptor.shape = shape
        descriptor.position = position
        descriptor.entity = zombie
        let character = backend.addCharacter(descriptor)
        XCTAssertNotNil(character)
        return character!
    }

    /// Moves the character with `velocity` for `seconds`, stepping the world
    /// after every move the way a game frame does.
    private func walk(_ character: JoltCharacter, _ backend: JoltPhysicsBackend, velocity: simd_float3, seconds: Float) {
        for _ in 0 ..< Int((seconds / step).rounded()) {
            character.move(velocity: velocity, deltaTime: step)
            backend.step(deltaTime: step)
        }
    }

    private func ball(at position: simd_float3, velocity: simd_float3 = .zero) -> PhysicsBodyDescriptor {
        PhysicsBodyDescriptor(
            motionType: .dynamic,
            collider: PhysicsColliderDescriptor(shape: .sphere(radius: 0.11), friction: 0.35, restitution: 0.6),
            mass: 0.6,
            position: position,
            linearVelocity: velocity
        )
    }

    // MARK: - Tests

    func testCapabilityAndFreeMovement() {
        let backend = makeBackend(wall: false)
        XCTAssertTrue(backend.capabilities.contains(.characterController))
        let character = makeCharacter(backend)
        XCTAssertEqual(character.groundState, .inAir) // no move yet: nothing sampled

        walk(character, backend, velocity: simd_float3(1, 0, 0), seconds: 1)
        XCTAssertEqual(character.position.x, 1, accuracy: 0.01)
        XCTAssertEqual(character.position.y, 0, accuracy: 0.005)
        XCTAssertEqual(character.position.z, 0, accuracy: 0.001)
        XCTAssertEqual(character.groundState, .onGround)

        character.move(by: simd_float3(0, 0, 0.5), deltaTime: step)
        XCTAssertEqual(character.position.z, 0.5, accuracy: 0.01)
    }

    func testStopsAtTheWallAndSlidesAlongIt() {
        let backend = makeBackend()
        let character = makeCharacter(backend)

        walk(character, backend, velocity: simd_float3(1, 0, 0), seconds: 2)
        // Wall face at 0.95, radius 0.2, padding 0.02: the centre stops at 0.73.
        XCTAssertEqual(character.position.x, 0.73, accuracy: 0.03)
        XCTAssertEqual(character.position.z, 0, accuracy: 0.01)
        // Pinned: the produced velocity is zero, whatever was asked.
        XCTAssertEqual(simd_length(character.velocity), 0, accuracy: 0.05)
        let touching = character.contacts().filter { $0.entity == JoltPhysicsBackend.environmentEntity && $0.normal.x < -0.9 }
        XCTAssertFalse(touching.isEmpty, "the wall should be among the contacts")

        // Diagonal into the wall: blocked along x, free along z.
        walk(character, backend, velocity: simd_float3(1, 0, 1), seconds: 1)
        XCTAssertEqual(character.position.x, 0.73, accuracy: 0.03)
        XCTAssertEqual(character.position.z, 1, accuracy: 0.05)
        XCTAssertEqual(character.velocity.x, 0, accuracy: 0.05)
        XCTAssertEqual(character.velocity.z, 1, accuracy: 0.05)
    }

    func testTriggerVolumesAreNotContactsAndManyContactsAreAllListed() {
        let backend = makeBackend(wall: false)
        // A trigger around the origin, and a ring of forty thin posts just
        // outside the shape (within its predictive contact distance).
        backend.didAddBody(entity: 500, descriptor: PhysicsBodyDescriptor(
            motionType: .static,
            collider: PhysicsColliderDescriptor(shape: .box(halfExtents: simd_float3(1, 1, 1)), isTrigger: true),
            position: simd_float3(0, 1, 0)
        ))
        var posts: [JoltEnvironmentBox] = []
        for index in 0 ..< 40 {
            let angle = Float(index) / 40 * 2 * .pi
            posts.append(JoltEnvironmentBox(
                center: simd_float3(cos(angle) * 0.27, 0.9, sin(angle) * 0.27),
                orientation: simd_quatf(angle: -angle, axis: simd_float3(0, 1, 0)),
                halfExtents: simd_float3(0.005, 0.9, 0.01)
            ))
        }
        backend.setEnvironmentBoxes(posts)
        backend.step(deltaTime: step)
        let character = makeCharacter(backend)
        character.move(velocity: .zero, deltaTime: step)

        let contacts = character.contacts()
        XCTAssertFalse(contacts.contains { $0.entity == 500 }, "the trigger is not geometry")
        XCTAssertGreaterThan(contacts.count, 32, "every post is listed, past the initial scratch")
        // The posts, and the floor it stands on (entity 1000): nothing else.
        XCTAssertTrue(contacts.allSatisfy { $0.entity == JoltPhysicsBackend.environmentEntity || $0.entity == 1000 })
    }

    func testZeroStrengthNeverPushes() throws {
        let backend = makeBackend(wall: false)
        backend.didAddBody(entity: 7, descriptor: ball(at: simd_float3(0.6, 0.11, 0)))
        backend.step(deltaTime: step)
        var descriptor = JoltCharacterDescriptor(radius: radius, height: height)
        descriptor.entity = zombie
        descriptor.maxStrength = 0
        let character = try XCTUnwrap(backend.addCharacter(descriptor))

        walk(character, backend, velocity: simd_float3(1.5, 0, 0), seconds: 0.5)
        // Blocked by a ball it cannot push (the inner body only nudges it).
        XCTAssertLessThan(backend.bodyState(for: 7)?.position.x ?? 0, 0.8)
        XCTAssertLessThan(character.position.x, 0.45)
    }

    func testCylinderKeepsItsHeightAgainstALowBox() {
        let backend = makeBackend(wall: false)
        backend.setEnvironmentBoxes([
            JoltEnvironmentBox(center: simd_float3(1.0, 0.075, 0), halfExtents: simd_float3(0.3, 0.075, 0.3)),
        ])
        backend.step(deltaTime: step)
        let character = makeCharacter(backend, shape: .cylinder)

        walk(character, backend, velocity: simd_float3(1, 0, 0), seconds: 2)
        // A 15 cm box is a wall to a character whose height the game owns.
        XCTAssertLessThan(character.position.x, 0.7 - 0.2)
        XCTAssertEqual(character.position.y, 0, accuracy: 0.005)
    }

    func testPushesADynamicBody() {
        let backend = makeBackend(wall: false)
        backend.didAddBody(entity: 7, descriptor: ball(at: simd_float3(0.6, 0.11, 0)))
        backend.step(deltaTime: step)
        let character = makeCharacter(backend)

        walk(character, backend, velocity: simd_float3(1.5, 0, 0), seconds: 1)
        let state = backend.bodyState(for: 7)
        XCTAssertNotNil(state)
        XCTAssertGreaterThan(state?.position.x ?? 0, 1.2, "the ball should have been pushed ahead")
        XCTAssertGreaterThan(character.position.x, 1.0, "the character keeps walking")
    }

    func testABallBouncesOffTheInnerBody() {
        let backend = makeBackend(wall: false)
        let character = makeCharacter(backend)
        backend.didAddBody(entity: 8, descriptor: ball(at: simd_float3(0, 1.0, -1.5), velocity: simd_float3(0, 0, 4)))

        walk(character, backend, velocity: .zero, seconds: 1)
        let state = backend.bodyState(for: 8)
        XCTAssertNotNil(state)
        XCTAssertLessThan(state?.position.z ?? 1, -0.2, "the ball should not pass through the character")
        XCTAssertEqual(character.position.z, 0, accuracy: 0.01, "the character is not moved by the ball")
    }

    func testRaysHitTheCharacterAsItsEntity() {
        let backend = makeBackend(wall: false)
        let character = makeCharacter(backend)
        backend.step(deltaTime: step)

        let hit = backend.raycast(
            PhysicsRay(origin: simd_float3(-2, 0.9, 0), direction: simd_float3(1, 0, 0), maxDistance: 10),
            filter: PhysicsQueryFilter()
        )
        XCTAssertNotNil(hit)
        XCTAssertEqual(hit?.entity, zombie)
        // The inner body is 90% of the outer radius.
        XCTAssertEqual(hit?.distance ?? 0, 2 - radius * 0.9, accuracy: 0.02)

        backend.removeCharacter(character)
        XCTAssertFalse(character.isValid)
        backend.step(deltaTime: step)
        let miss = backend.raycast(
            PhysicsRay(origin: simd_float3(-2, 0.9, 0), direction: simd_float3(1, 0, 0), maxDistance: 10),
            filter: PhysicsQueryFilter()
        )
        XCTAssertNil(miss)
    }

    func testRemovalRestoresTheBodyCountAndSilencesTheHandle() {
        let backend = makeBackend(wall: false)
        let before = backend.bodyCount
        let character = makeCharacter(backend)
        XCTAssertEqual(backend.bodyCount, before + 1, "the inner body is listed")

        backend.removeCharacter(character)
        XCTAssertEqual(backend.bodyCount, before)
        character.move(velocity: simd_float3(1, 0, 0), deltaTime: step)
        XCTAssertEqual(character.position, .zero)
        XCTAssertTrue(character.contacts().isEmpty)
        backend.removeCharacter(character) // twice is harmless
    }

    func testTeleportNextToTheWallThenPushingIntoItStaysOut() {
        let backend = makeBackend()
        let character = makeCharacter(backend)

        character.teleport(to: simd_float3(0.7, 0, 0.5))
        XCTAssertEqual(character.position.x, 0.7, accuracy: 0.001)
        walk(character, backend, velocity: simd_float3(2, 0, 0), seconds: 0.5)
        XCTAssertEqual(character.position.x, 0.73, accuracy: 0.03)
        XCTAssertEqual(character.position.z, 0.5, accuracy: 0.01)
    }

    func testCharactersDoNotOutliveTheWorld() {
        // The world destructor must release the character (and its inner
        // body) before sweeping the bodies; a crash here is the failure.
        let character: JoltCharacter = {
            let backend = makeBackend()
            let character = makeCharacter(backend)
            walk(character, backend, velocity: simd_float3(0, 0, 1), seconds: 0.2)
            return character // the backend, and its world, die here
        }()
        XCTAssertTrue(character.isValid, "the handle is not told; the owner must not use it")
    }
}
