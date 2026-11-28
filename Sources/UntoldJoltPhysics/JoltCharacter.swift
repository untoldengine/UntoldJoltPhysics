//
//  JoltCharacter.swift
//  UntoldJoltPhysics
//
//  A character controller — Jolt's CharacterVirtual — through the plugin's
//  side channel. The engine's physics seam only names the capability, so a
//  game creates the character here and moves it itself every frame: it hands
//  over the velocity its animation produced, the controller collides and
//  slides that motion against the environment and the rigid bodies (pushing
//  dynamic ones), and the game reads the corrected position back. The world
//  step never moves a character.
//

import CJoltBridge
import simd
import UntoldEngine

public struct JoltCharacterDescriptor: Sendable {
    public enum Shape: Sendable {
        /// Rounded ends: slides smoothly over small steps and edges.
        case capsule
        /// A flat base: every side contact is horizontal, so a character whose
        /// height the game pins never climbs the edge of a low object.
        case cylinder
    }

    public var shape: Shape = .capsule
    public var radius: Float
    /// Total height, base to top (a capsule needs at least twice the radius).
    public var height: Float
    /// The base of the shape — the feet — not its centre.
    public var position: SIMD3<Float> = .zero
    public var rotation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    public var layer: UInt32 = 0
    /// Reported by contacts with, and ray hits on, the character's inner body.
    public var entity: EntityID = JoltPhysicsBackend.environmentEntity
    /// Presses down on what the character stands on (with a non-zero
    /// gravity); 0 never does.
    public var mass: Float = 70
    /// The most force, in newtons, a pushed dynamic body receives; 0 never
    /// pushes.
    public var maxStrength: Float = 100
    /// The distance kept from every surface: the character stops this far
    /// short. Must be positive (0 or less falls back to 2 cm).
    public var padding: Float = 0.02
    /// Must be positive (0 or less falls back to 10 cm; at 0 the character
    /// would stick).
    public var predictiveContactDistance: Float = 0.1
    /// Contacts steeper than this, in radians from the vertical, are walls;
    /// 0 makes every contact a wall.
    public var maxSlopeAngle: Float = 50 * .pi / 180
    /// The fraction of a penetration resolved per move (0 or less falls
    /// back to 1).
    public var penetrationRecoverySpeed: Float = 1
    /// A kinematic body inside the shape: dynamic bodies bounce off the
    /// character and rays hit it. It carries `entity`, is never read back
    /// and goes away with the character.
    public var innerBody = true
    /// Must be positive (0 or less falls back to 0.9).
    public var innerBodyFraction: Float = 0.9
    /// Whether the character's own collision ignores dynamic bodies. On, it
    /// neither blocks on them nor pushes them: they meet only its inner
    /// body, a plain kinematic body whose contacts the world reports, so a
    /// game hears every hit. Off, the controller catches and pushes them
    /// itself, and those contacts reach no listener.
    public var ignoresDynamicBodies = false
    /// Whether dynamic bodies may shove the character. Off, a ball resting
    /// against it or hitting it never moves it, while the character still
    /// pushes the ball; kinematic bodies (a tracked hand) push it either way.
    public var pushedByDynamicBodies = false
    /// Metres the character may snap down to stay on a floor it walks off;
    /// 0 turns it off (the game owns the height).
    public var stickToFloorStepDown: Float = 0
    /// Metres the character may step up onto; 0 turns stair walking off.
    public var walkStairsStepUp: Float = 0

    public init(radius: Float, height: Float) {
        self.radius = radius
        self.height = height
    }
}

public enum JoltCharacterGroundState: Sendable {
    case onGround
    case onSteepGround
    case notSupported
    case inAir
}

/// A contact the character's last move found. Trigger volumes are never
/// listed: the character passes through them.
public struct JoltCharacterContact: Sendable {
    /// The other body's entity; `JoltPhysicsBackend.environmentEntity` for
    /// environment geometry.
    public var entity: EntityID
    public var position: SIMD3<Float>
    /// Points towards the character.
    public var normal: SIMD3<Float>
    /// At most zero when touching or penetrating; positive for a contact
    /// predicted ahead of the character.
    public var distance: Float
    public var isDynamic: Bool
    /// Whether the last move actually collided with it.
    public var hadCollision: Bool
}

/// A character in the world; remove it with `JoltPhysicsBackend.removeCharacter`.
/// Every call is frame-thread only, between world steps.
public final class JoltCharacter: @unchecked Sendable {
    private(set) var handle: OpaquePointer?
    public let entity: EntityID
    public let descriptor: JoltCharacterDescriptor
    private var contactScratch: [ujolt_character_contact]

    init(handle: OpaquePointer, descriptor: JoltCharacterDescriptor) {
        self.handle = handle
        self.descriptor = descriptor
        entity = descriptor.entity
        contactScratch = Array(repeating: ujolt_character_contact(), count: 32)
    }

    /// False once the character has been removed; every call is then a no-op.
    public var isValid: Bool { handle != nil }

    /// The base of the shape, in world space.
    public var position: SIMD3<Float> {
        guard let handle else { return .zero }
        var out: (Float, Float, Float) = (0, 0, 0)
        withUnsafeMutablePointer(to: &out) { p in
            p.withMemoryRebound(to: Float.self, capacity: 3) { ujolt_character_get_position(handle, $0) }
        }
        return SIMD3<Float>(out.0, out.1, out.2)
    }

    /// The velocity the last move actually produced — its displacement over
    /// its `deltaTime`, after sliding and stopping — not the one asked for.
    /// Zero before any move.
    public var velocity: SIMD3<Float> {
        guard let handle else { return .zero }
        var out: (Float, Float, Float) = (0, 0, 0)
        withUnsafeMutablePointer(to: &out) { p in
            p.withMemoryRebound(to: Float.self, capacity: 3) { ujolt_character_get_velocity(handle, $0) }
        }
        return SIMD3<Float>(out.0, out.1, out.2)
    }

    public var groundState: JoltCharacterGroundState {
        guard let handle else { return .inAir }
        switch ujolt_character_ground_state(handle) {
        case Int32(UJOLT_GROUND_ON_GROUND.rawValue): return .onGround
        case Int32(UJOLT_GROUND_ON_STEEP_GROUND.rawValue): return .onSteepGround
        case Int32(UJOLT_GROUND_NOT_SUPPORTED.rawValue): return .notSupported
        default: return .inAir
        }
    }

    /// Moves by `velocity * deltaTime`, colliding and sliding. `gravity` only
    /// presses on whatever the character stands on; it is never added to the
    /// character's own motion, so the default keeps the height where the
    /// game put it.
    public func move(velocity: SIMD3<Float>, deltaTime: Float, gravity: SIMD3<Float> = .zero) {
        guard let handle, deltaTime > 0 else { return }
        var v: (Float, Float, Float) = (velocity.x, velocity.y, velocity.z)
        var g: (Float, Float, Float) = (gravity.x, gravity.y, gravity.z)
        withUnsafePointer(to: &v) { vp in
            withUnsafePointer(to: &g) { gp in
                vp.withMemoryRebound(to: Float.self, capacity: 3) { vf in
                    gp.withMemoryRebound(to: Float.self, capacity: 3) { gf in
                        ujolt_character_move(handle, vf, deltaTime, gf)
                    }
                }
            }
        }
    }

    /// Moves by `delta` over `deltaTime` (the motion an animation produced
    /// this frame), colliding and sliding.
    public func move(by delta: SIMD3<Float>, deltaTime: Float, gravity: SIMD3<Float> = .zero) {
        guard deltaTime > 0 else { return }
        move(velocity: delta / deltaTime, deltaTime: deltaTime, gravity: gravity)
    }

    /// Teleports the base; the contacts are recomputed at the new place.
    public func teleport(to position: SIMD3<Float>) {
        guard let handle else { return }
        var p: (Float, Float, Float) = (position.x, position.y, position.z)
        withUnsafePointer(to: &p) { pp in
            pp.withMemoryRebound(to: Float.self, capacity: 3) { ujolt_character_set_position(handle, $0) }
        }
    }

    public func setRotation(_ rotation: simd_quatf) {
        guard let handle else { return }
        var q: (Float, Float, Float, Float) = (rotation.imag.x, rotation.imag.y, rotation.imag.z, rotation.real)
        withUnsafePointer(to: &q) { qp in
            qp.withMemoryRebound(to: Float.self, capacity: 4) { ujolt_character_set_rotation(handle, $0) }
        }
    }

    /// The contacts the last move found (touching, penetrating and predicted;
    /// never trigger volumes).
    public func contacts() -> [JoltCharacterContact] {
        guard let handle else { return [] }
        var count = contactScratch.withUnsafeMutableBufferPointer { buffer in
            Int(ujolt_character_contacts(handle, buffer.baseAddress, UInt32(buffer.count)))
        }
        if count > contactScratch.count {
            // More than the scratch holds: grow it and ask again.
            contactScratch = Array(repeating: ujolt_character_contact(), count: count)
            count = contactScratch.withUnsafeMutableBufferPointer { buffer in
                Int(ujolt_character_contacts(handle, buffer.baseAddress, UInt32(buffer.count)))
            }
        }
        return (0 ..< min(count, contactScratch.count)).map { index in
            let c = contactScratch[index]
            return JoltCharacterContact(
                entity: EntityID(c.user_data),
                position: SIMD3<Float>(c.position.0, c.position.1, c.position.2),
                normal: SIMD3<Float>(c.normal.0, c.normal.1, c.normal.2),
                distance: c.distance,
                isDynamic: c.is_dynamic != 0,
                hadCollision: c.had_collision != 0
            )
        }
    }

    func invalidate() {
        handle = nil
    }
}

extension JoltPhysicsBackend {
    /// Adds a character controller; nil for a degenerate shape. Frame thread.
    public func addCharacter(_ descriptor: JoltCharacterDescriptor) -> JoltCharacter? {
        guard descriptor.radius > 0, descriptor.height > 0 else { return nil }
        var desc = ujolt_character_desc()
        desc.shape = descriptor.shape == .cylinder ? UJOLT_CHARACTER_CYLINDER : UJOLT_CHARACTER_CAPSULE
        desc.radius = descriptor.radius
        desc.height = descriptor.height
        desc.position = (descriptor.position.x, descriptor.position.y, descriptor.position.z)
        let q = descriptor.rotation
        desc.rotation = (q.imag.x, q.imag.y, q.imag.z, q.real)
        desc.layer = descriptor.layer
        desc.user_data = UInt64(descriptor.entity)
        desc.mass = descriptor.mass
        desc.max_strength = descriptor.maxStrength
        desc.padding = descriptor.padding
        desc.predictive_contact_distance = descriptor.predictiveContactDistance
        desc.max_slope_degrees = descriptor.maxSlopeAngle * 180 / .pi
        desc.penetration_recovery_speed = descriptor.penetrationRecoverySpeed
        desc.inner_body = descriptor.innerBody ? 1 : 0
        desc.inner_body_fraction = descriptor.innerBodyFraction
        desc.pushed_by_dynamic_bodies = descriptor.pushedByDynamicBodies ? 1 : 0
        desc.ignores_dynamic_bodies = descriptor.ignoresDynamicBodies ? 1 : 0
        desc.stick_to_floor_step_down = descriptor.stickToFloorStepDown
        desc.walk_stairs_step_up = descriptor.walkStairsStepUp
        guard let handle = ujolt_world_add_character(worldHandle, &desc) else { return nil }
        return JoltCharacter(handle: handle, descriptor: descriptor)
    }

    /// Removes the character and its inner body. Frame thread.
    public func removeCharacter(_ character: JoltCharacter) {
        guard let handle = character.handle else { return }
        ujolt_world_remove_character(worldHandle, handle)
        character.invalidate()
    }
}
