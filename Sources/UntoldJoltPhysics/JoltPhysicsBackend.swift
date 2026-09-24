//
//  JoltPhysicsBackend.swift
//  UntoldJoltPhysics
//
//  The engine's PhysicsBackend implemented on Jolt Physics. A thin adapter:
//  the engine's descriptors become Jolt bodies through the C shim, kinematic
//  targets are forwarded as kinematic moves, transforms are read back for
//  the bodies Jolt moved, and the shim's buffered contact/activation events
//  become the engine's PhysicsEventSink calls.
//

import Foundation
import simd
import UntoldEngine
import CJoltBridge

/// Tunables for the Jolt world. Defaults suit a room-scale scene.
public struct JoltWorldSettings: Sendable {
    public var maxBodies: UInt32 = 4096
    public var maxBodyPairs: UInt32 = 4096
    public var maxContactConstraints: UInt32 = 2048
    public var tempAllocatorBytes: UInt32 = 16 * 1024 * 1024
    /// Jolt worker threads. `nil` uses all cores but one; 0 runs the
    /// simulation single-threaded on the calling thread (deterministic).
    public var workerThreads: Int32? = nil
    /// Also deliver `.persisted` contact events every step for resting
    /// contacts (off by default: it floods the event buffers).
    public var reportPersistedContacts = false
    /// Read-back and event scratch capacity per substep.
    public var readbackCapacity = 4096
    /// A kinematic target farther than this (metres) from the body's current
    /// position teleports it instead of moving it, so a tracked hand
    /// re-appearing far away cannot launch what it lands on. `nil` = unlimited.
    public var maxKinematicStep: Float? = nil
    /// Ceiling on the velocity a kinematic move may carry (m/s): a target
    /// within `maxKinematicStep` that arrives fast — a tracking glitch, a
    /// stalled frame — approaches at this speed over a few substeps instead
    /// of striking at the implied one. `nil` = unlimited.
    public var maxKinematicSpeed: Float? = nil
    /// Dynamic bodies use Jolt's continuous collision (LinearCast), so a fast
    /// body cannot pass through thin geometry. Costs a little per body.
    public var continuousCollision = true
    /// Non-trigger contacts closing slower than this (m/s) are not reported
    /// as `.began`: a resting contact re-established after a sleep or a
    /// geometry rebuild is not an impact. Keep it above one substep of
    /// gravity (9.8 / 60 ≈ 0.16 m/s), which is what a just-woken resting body
    /// closes at. `.ended` follows only a reported `.began`. 0 reports
    /// everything.
    public var minContactSpeed: Float = 0.3

    public init() {}
}

/// A static box that is part of the environment rather than of an entity —
/// detected real-world surfaces in a mixed-reality app. Contacts with it
/// report `JoltPhysicsBackend.environmentEntity`.
public struct JoltEnvironmentBox: Equatable, Sendable {
    public var center: simd_float3
    public var orientation: simd_quatf
    public var halfExtents: simd_float3
    public var friction: Float
    public var restitution: Float

    public init(
        center: simd_float3,
        orientation: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1),
        halfExtents: simd_float3,
        friction: Float = 0.5,
        restitution: Float = 0.0
    ) {
        self.center = center
        self.orientation = orientation
        self.halfExtents = halfExtents
        self.friction = friction
        self.restitution = restitution
    }
}

public final class JoltPhysicsBackend: PhysicsBackend, @unchecked Sendable {
    public let id = JoltPhysicsPluginContract.backendID
    public let capabilities: PhysicsCapabilities = [.collisions, .triggers, .raycast, .characterController]
    /// Entity reported for contacts with (and raycast hits on) environment
    /// geometry, which belongs to no entity. The engine's null entity, so no
    /// script can ever be triggered by it.
    public static let environmentEntity: EntityID = .invalid

    private let world: OpaquePointer
    /// The bridge world, for the module's extensions (soft bodies, characters).
    var worldHandle: OpaquePointer { world }
    private let settings: JoltWorldSettings

    // Frame-thread state (the coordinator contract: every protocol call
    // arrives from the frame thread in order).
    private var bodyByEntity: [EntityID: ujolt_body_id] = [:]
    private var entityByBody: [ujolt_body_id: EntityID] = [:]
    private var environmentBodies: [ujolt_body_id] = []
    private var environmentBoxes: [JoltEnvironmentBox] = []
    private let environmentLock = NSLock()
    private var pendingEnvironment: [JoltEnvironmentBox]?
    private var changedScratch: [ujolt_body_id]
    private var contactScratch: [ujolt_contact_event]
    private var activationScratch: [ujolt_activation_event]

    public init(settings: JoltWorldSettings = JoltWorldSettings()) {
        self.settings = settings
        var desc = ujolt_world_desc()
        desc.max_bodies = settings.maxBodies
        desc.max_body_pairs = settings.maxBodyPairs
        desc.max_contact_constraints = settings.maxContactConstraints
        desc.temp_allocator_bytes = settings.tempAllocatorBytes
        desc.worker_threads = settings.workerThreads ?? -1
        desc.report_persisted_contacts = settings.reportPersistedContacts ? 1 : 0
        desc.gravity = (0, -9.8, 0)
        desc.max_kinematic_step = settings.maxKinematicStep ?? 0
        desc.max_kinematic_speed = settings.maxKinematicSpeed ?? 0
        desc.continuous_collision = settings.continuousCollision ? 1 : 0
        desc.min_contact_speed = settings.minContactSpeed
        guard let world = ujolt_world_create(&desc) else {
            fatalError("UntoldJoltPhysics: could not create the Jolt world")
        }
        self.world = world
        // Every body can be active at once: the read-back scratch must hold
        // them all, whatever readbackCapacity says.
        let capacity = max(settings.readbackCapacity, Int(settings.maxBodies))
        changedScratch = Array(repeating: 0, count: capacity)
        contactScratch = Array(repeating: ujolt_contact_event(), count: settings.readbackCapacity)
        activationScratch = Array(repeating: ujolt_activation_event(), count: settings.readbackCapacity)
    }

    deinit {
        ujolt_world_destroy(world)
    }

    /// Number of bodies currently simulated (diagnostics).
    public var bodyCount: Int {
        Int(ujolt_world_body_count(world))
    }

    // MARK: - PhysicsBackend

    public func configure(_ config: PhysicsWorldConfiguration) {
        var gravity = (config.gravity.x, config.gravity.y, config.gravity.z)
        withUnsafePointer(to: &gravity) { pointer in
            pointer.withMemoryRebound(to: Float.self, capacity: 3) { ujolt_world_set_gravity(world, $0) }
        }
        if config.collisionLayerMatrix.isEmpty {
            ujolt_world_set_layer_matrix(world, nil, 0)
        } else {
            config.collisionLayerMatrix.withUnsafeBufferPointer { masks in
                ujolt_world_set_layer_matrix(world, masks.baseAddress, UInt32(masks.count))
            }
        }
    }

    public func didAddBody(entity: EntityID, descriptor: PhysicsBodyDescriptor) {
        if let existing = bodyByEntity[entity] {
            ujolt_world_remove_body(world, existing)
            entityByBody.removeValue(forKey: existing)
            bodyByEntity.removeValue(forKey: entity)
        }

        var desc = ujolt_body_desc()
        var hullPoints: [Float] = []
        switch descriptor.collider.shape {
        case let .sphere(radius):
            desc.shape = UJOLT_SHAPE_SPHERE
            desc.radius = radius
        case let .box(halfExtents):
            desc.shape = UJOLT_SHAPE_BOX
            desc.half_extents = (halfExtents.x, halfExtents.y, halfExtents.z)
        case let .capsule(radius, height):
            desc.shape = UJOLT_SHAPE_CAPSULE
            desc.radius = radius
            desc.half_height = height * 0.5
        case let .cylinder(radius, height):
            desc.shape = UJOLT_SHAPE_CYLINDER
            desc.radius = radius
            desc.half_height = height * 0.5
        case let .convexHull(vertices):
            desc.shape = UJOLT_SHAPE_CONVEX_HULL
            hullPoints.reserveCapacity(vertices.count * 3)
            for vertex in vertices {
                hullPoints.append(vertex.x)
                hullPoints.append(vertex.y)
                hullPoints.append(vertex.z)
            }
            desc.hull_point_count = UInt32(vertices.count)
        }
        let offset = descriptor.collider.localOffset
        desc.local_offset = (offset.x, offset.y, offset.z)
        desc.friction = descriptor.collider.friction
        desc.restitution = descriptor.collider.restitution
        desc.is_sensor = descriptor.collider.isTrigger ? 1 : 0
        switch descriptor.motionType {
        case .static: desc.motion = UJOLT_MOTION_STATIC
        case .kinematic: desc.motion = UJOLT_MOTION_KINEMATIC
        case .dynamic: desc.motion = UJOLT_MOTION_DYNAMIC
        }
        desc.mass = descriptor.mass
        desc.layer = descriptor.layer
        desc.gravity_factor = descriptor.gravityScale
        desc.position = (descriptor.position.x, descriptor.position.y, descriptor.position.z)
        let q = descriptor.orientation
        desc.rotation = (q.imag.x, q.imag.y, q.imag.z, q.real)
        desc.linear_velocity = (descriptor.linearVelocity.x, descriptor.linearVelocity.y, descriptor.linearVelocity.z)
        desc.angular_velocity = (descriptor.angularVelocity.x, descriptor.angularVelocity.y, descriptor.angularVelocity.z)
        desc.user_data = UInt64(entity)

        let body: ujolt_body_id = hullPoints.withUnsafeBufferPointer { points in
            desc.hull_points = points.baseAddress
            return ujolt_world_add_body(world, &desc)
        }
        guard body != UJOLT_INVALID_BODY else {
            // The coordinator will not retry: say so, loudly enough to notice.
            print("UntoldJoltPhysics: Jolt refused a body for entity \(entity) (\(descriptor.collider.shape)) — degenerate shape or maxBodies reached")
            return
        }
        bodyByEntity[entity] = body
        entityByBody[body] = entity
    }

    public func didRemoveBody(entity: EntityID) {
        guard let body = bodyByEntity.removeValue(forKey: entity) else { return }
        entityByBody.removeValue(forKey: body)
        ujolt_world_remove_body(world, body)
    }

    public func writeKinematicTargets(_ batch: PhysicsBodyWriteBatch) {
        for index in 0 ..< batch.entities.count {
            guard let body = bodyByEntity[batch.entities[index]] else { continue }
            let transform = batch.transforms[index]
            var position = (transform.position.x, transform.position.y, transform.position.z)
            var rotation = (transform.orientation.imag.x, transform.orientation.imag.y,
                            transform.orientation.imag.z, transform.orientation.real)
            withUnsafePointer(to: &position) { p in
                withUnsafePointer(to: &rotation) { r in
                    p.withMemoryRebound(to: Float.self, capacity: 3) { pf in
                        r.withMemoryRebound(to: Float.self, capacity: 4) { rf in
                            ujolt_world_set_kinematic_target(world, body, pf, rf)
                        }
                    }
                }
            }
        }
    }

    public func step(deltaTime: Float) {
        applyPendingEnvironment()
        ujolt_world_step(world, deltaTime, 1)
    }

    public func readActiveTransforms(into batch: PhysicsTransformReadBatch) -> Int {
        let count = changedScratch.withUnsafeMutableBufferPointer { ids in
            Int(ujolt_world_changed_bodies(world, ids.baseAddress, UInt32(ids.count)))
        }
        var written = 0
        var position: (Float, Float, Float) = (0, 0, 0)
        var rotation: (Float, Float, Float, Float) = (0, 0, 0, 1)
        for index in 0 ..< count {
            guard written < batch.capacity else { break }
            let body = changedScratch[index]
            guard let entity = entityByBody[body] else { continue }
            withUnsafeMutablePointer(to: &position) { p in
                withUnsafeMutablePointer(to: &rotation) { r in
                    p.withMemoryRebound(to: Float.self, capacity: 3) { pf in
                        r.withMemoryRebound(to: Float.self, capacity: 4) { rf in
                            ujolt_world_get_transform(world, body, pf, rf)
                        }
                    }
                }
            }
            batch.entities[written] = entity
            batch.transforms[written] = PhysicsBodyTransform(
                position: simd_float3(position.0, position.1, position.2),
                orientation: simd_quatf(ix: rotation.0, iy: rotation.1, iz: rotation.2, r: rotation.3)
            )
            written += 1
        }
        return written
    }

    public func drainEvents(into sink: any PhysicsEventSink) {
        var dropped: UInt32 = 0
        let contactCount = contactScratch.withUnsafeMutableBufferPointer { buffer in
            Int(ujolt_world_drain_contacts(world, buffer.baseAddress, UInt32(buffer.count), &dropped))
        }
        var totalDropped = Int(dropped)
        for index in 0 ..< contactCount {
            let event = contactScratch[index]
            let entityA = EntityID(event.user_a)
            let entityB = EntityID(event.user_b)
            if event.sensor_a != 0 || event.sensor_b != 0 {
                // Sensor contacts are the engine's trigger events.
                guard event.phase != UJOLT_CONTACT_PERSISTED else { continue }
                let (trigger, other) = event.sensor_a != 0 ? (entityA, entityB) : (entityB, entityA)
                sink.receiveTrigger(PhysicsTriggerEvent(
                    phase: event.phase == UJOLT_CONTACT_ADDED ? .entered : .exited,
                    triggerEntity: trigger,
                    otherEntity: other
                ))
                continue
            }
            let phase: PhysicsContactPhase
            switch event.phase {
            case UJOLT_CONTACT_ADDED: phase = .began
            case UJOLT_CONTACT_PERSISTED: phase = .persisted
            default: phase = .ended
            }
            sink.receiveContact(PhysicsContactEvent(
                phase: phase,
                entityA: entityA,
                entityB: entityB,
                position: simd_float3(event.position.0, event.position.1, event.position.2),
                normal: simd_float3(event.normal.0, event.normal.1, event.normal.2),
                impulse: event.impulse
            ))
        }

        let activationCount = activationScratch.withUnsafeMutableBufferPointer { buffer in
            Int(ujolt_world_drain_activations(world, buffer.baseAddress, UInt32(buffer.count), &dropped))
        }
        totalDropped += Int(dropped)
        for index in 0 ..< activationCount {
            let event = activationScratch[index]
            sink.receiveActivation(PhysicsBodyActivationEvent(
                entity: EntityID(event.user_data),
                isActive: event.is_active != 0
            ))
        }
        if totalDropped > 0 {
            sink.reportDroppedEvents(count: totalDropped)
        }
    }

    public func raycast(_ ray: PhysicsRay, filter: PhysicsQueryFilter) -> PhysicsRayHit? {
        let maxDistance = min(ray.maxDistance, 1.0e6)
        var origin = (ray.origin.x, ray.origin.y, ray.origin.z)
        var direction = (ray.direction.x, ray.direction.y, ray.direction.z)
        var hit = ujolt_ray_hit()
        let excluded = Array(filter.excludedEntities).map { UInt64($0) }
        let found: Int32 = withUnsafePointer(to: &origin) { o in
            withUnsafePointer(to: &direction) { d in
                o.withMemoryRebound(to: Float.self, capacity: 3) { of in
                    d.withMemoryRebound(to: Float.self, capacity: 3) { df in
                        excluded.withUnsafeBufferPointer { ex in
                            ujolt_world_cast_ray(world, of, df, maxDistance, filter.layerMask,
                                                 ex.baseAddress, UInt32(ex.count), &hit)
                        }
                    }
                }
            }
        }
        guard found != 0 else { return nil }
        return PhysicsRayHit(
            entity: EntityID(hit.user_data),
            position: simd_float3(hit.position.0, hit.position.1, hit.position.2),
            normal: simd_float3(hit.normal.0, hit.normal.1, hit.normal.2),
            distance: hit.distance
        )
    }

    // MARK: - Plugin-owned extras

    /// Replaces the environment geometry: static boxes that belong to no
    /// entity (e.g. ARKit-detected floors, walls and tables). Contacts with
    /// them report `JoltPhysicsBackend.environmentEntity`. Safe from any
    /// thread: the change is queued and applied on the frame thread at the
    /// start of the next `step`.
    public func setEnvironmentBoxes(_ boxes: [JoltEnvironmentBox]) {
        environmentLock.withLock { pendingEnvironment = boxes }
    }

    private func applyPendingEnvironment() {
        guard let boxes = environmentLock.withLock({ () -> [JoltEnvironmentBox]? in
            defer { pendingEnvironment = nil }
            return pendingEnvironment
        }) else { return }
        // Keep the Jolt bodies of boxes that did not change (plane streams
        // re-send everything on every update): only the changed ones are
        // rebuilt, so resting contacts and sleep state survive.
        var unclaimed = Array(zip(environmentBoxes, environmentBodies))
        var keptBoxes: [JoltEnvironmentBox] = []
        var keptBodies: [ujolt_body_id] = []
        var newBoxes: [JoltEnvironmentBox] = []
        for box in boxes {
            if let index = unclaimed.firstIndex(where: { $0.0 == box }) {
                let (kept, body) = unclaimed.remove(at: index)
                keptBoxes.append(kept)
                keptBodies.append(body)
            } else {
                newBoxes.append(box)
            }
        }
        // Jolt does not wake a sleeping body when the surface under it goes
        // away; wake whatever rests on a removed box before removing it.
        for (box, body) in unclaimed {
            let reach = simd_length(box.halfExtents) + 0.1
            var lower = (box.center.x - reach, box.center.y - reach, box.center.z - reach)
            var upper = (box.center.x + reach, box.center.y + reach, box.center.z + reach)
            withUnsafePointer(to: &lower) { l in
                withUnsafePointer(to: &upper) { u in
                    l.withMemoryRebound(to: Float.self, capacity: 3) { lf in
                        u.withMemoryRebound(to: Float.self, capacity: 3) { uf in
                            ujolt_world_activate_in_box(world, lf, uf)
                        }
                    }
                }
            }
            ujolt_world_remove_body(world, body)
        }
        environmentBoxes = keptBoxes
        environmentBodies = keptBodies
        for box in newBoxes {
            var desc = ujolt_body_desc()
            desc.shape = UJOLT_SHAPE_BOX
            desc.half_extents = (box.halfExtents.x, box.halfExtents.y, box.halfExtents.z)
            desc.friction = box.friction
            desc.restitution = box.restitution
            desc.motion = UJOLT_MOTION_STATIC
            desc.gravity_factor = 1
            desc.position = (box.center.x, box.center.y, box.center.z)
            desc.rotation = (box.orientation.imag.x, box.orientation.imag.y, box.orientation.imag.z, box.orientation.real)
            desc.user_data = UInt64(Self.environmentEntity)
            let body = ujolt_world_add_body(world, &desc)
            if body != UJOLT_INVALID_BODY {
                environmentBoxes.append(box)
                environmentBodies.append(body)
            }
        }
    }

    /// Number of environment boxes currently in the world (diagnostics;
    /// reflects the last applied set, not a pending one).
    public var environmentBodyCount: Int {
        environmentBodies.count
    }

    /// Position and velocity of a simulated body (game-logic read-back).
    public func bodyState(for entity: EntityID) -> (position: simd_float3, velocity: simd_float3)? {
        guard let body = bodyByEntity[entity] else { return nil }
        var position: (Float, Float, Float) = (0, 0, 0)
        var rotation: (Float, Float, Float, Float) = (0, 0, 0, 1)
        var linear: (Float, Float, Float) = (0, 0, 0)
        var angular: (Float, Float, Float) = (0, 0, 0)
        withUnsafeMutablePointer(to: &position) { p in
            withUnsafeMutablePointer(to: &rotation) { r in
                p.withMemoryRebound(to: Float.self, capacity: 3) { pf in
                    r.withMemoryRebound(to: Float.self, capacity: 4) { rf in
                        ujolt_world_get_transform(world, body, pf, rf)
                    }
                }
            }
        }
        withUnsafeMutablePointer(to: &linear) { l in
            withUnsafeMutablePointer(to: &angular) { a in
                l.withMemoryRebound(to: Float.self, capacity: 3) { lf in
                    a.withMemoryRebound(to: Float.self, capacity: 3) { af in
                        ujolt_world_get_velocity(world, body, lf, af)
                    }
                }
            }
        }
        return (simd_float3(position.0, position.1, position.2), simd_float3(linear.0, linear.1, linear.2))
    }

    /// Teleports a simulated body and sets its velocity (e.g. a ball reset).
    /// Returns false if the entity has no body right now.
    @discardableResult
    public func resetBody(entity: EntityID, position: simd_float3, velocity: simd_float3) -> Bool {
        guard let body = bodyByEntity[entity] else { return false }
        var p = (position.x, position.y, position.z)
        var r: (Float, Float, Float, Float) = (0, 0, 0, 1)
        var l = (velocity.x, velocity.y, velocity.z)
        var a: (Float, Float, Float) = (0, 0, 0)
        withUnsafePointer(to: &p) { pp in
            withUnsafePointer(to: &r) { rp in
                pp.withMemoryRebound(to: Float.self, capacity: 3) { pf in
                    rp.withMemoryRebound(to: Float.self, capacity: 4) { rf in
                        ujolt_world_set_transform(world, body, pf, rf)
                    }
                }
            }
        }
        withUnsafePointer(to: &l) { lp in
            withUnsafePointer(to: &a) { ap in
                lp.withMemoryRebound(to: Float.self, capacity: 3) { lf in
                    ap.withMemoryRebound(to: Float.self, capacity: 3) { af in
                        ujolt_world_set_velocity(world, body, lf, af)
                    }
                }
            }
        }
        return true
    }

    public func isBodyActive(entity: EntityID) -> Bool {
        guard let body = bodyByEntity[entity] else { return false }
        return ujolt_world_body_is_active(world, body) != 0
    }
}
