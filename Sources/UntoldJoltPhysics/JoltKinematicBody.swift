//
//  JoltKinematicBody.swift
//  UntoldJoltPhysics
//
//  Kinematic colliders that are not engine entities: a capsule or a convex
//  hull per bone of an animated character, moved every frame to follow the
//  skeleton, for cloth and other soft bodies to collide with. Same side
//  channel as the soft bodies.
//

import CJoltBridge
import simd
import UntoldEngine

/// A kinematic body in the world; remove it with `removeKinematicBody`.
public final class JoltKinematicBody: @unchecked Sendable {
    let id: ujolt_body_id

    init(id: ujolt_body_id) {
        self.id = id
    }
}

public extension JoltPhysicsBackend {
    /// Adds a kinematic capsule (axis along its local y, `height` the total
    /// length including the caps). Frame thread.
    func addKinematicCapsule(
        radius: Float, height: Float, position: simd_float3, rotation: simd_quatf,
        friction: Float = 0.5, layer: UInt32 = 0
    ) -> JoltKinematicBody? {
        guard radius > 0, height >= 2 * radius else { return nil }
        var desc = ujolt_body_desc()
        desc.shape = UJOLT_SHAPE_CAPSULE
        desc.radius = radius
        desc.half_height = max(height * 0.5 - radius, 0)
        desc.friction = friction
        desc.restitution = 0
        desc.motion = UJOLT_MOTION_KINEMATIC
        desc.gravity_factor = 0
        desc.layer = layer
        desc.position = (position.x, position.y, position.z)
        desc.rotation = (rotation.imag.x, rotation.imag.y, rotation.imag.z, rotation.real)
        desc.user_data = UInt64(Self.environmentEntity)
        let id = ujolt_world_add_body(worldHandle, &desc)
        guard id != UJOLT_INVALID_BODY else { return nil }
        return JoltKinematicBody(id: id)
    }

    /// Adds a kinematic convex hull of `points` (body-local; at least four
    /// not all coplanar, a few dozen at most is plenty). Frame thread.
    func addKinematicConvexHull(
        points: [simd_float3], position: simd_float3, rotation: simd_quatf,
        friction: Float = 0.5, layer: UInt32 = 0
    ) -> JoltKinematicBody? {
        guard points.count >= 4, points.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else { return nil }
        var flat: [Float] = []
        flat.reserveCapacity(points.count * 3)
        for p in points {
            flat.append(p.x)
            flat.append(p.y)
            flat.append(p.z)
        }
        return flat.withUnsafeBufferPointer { buffer -> JoltKinematicBody? in
            var desc = ujolt_body_desc()
            desc.shape = UJOLT_SHAPE_CONVEX_HULL
            desc.hull_points = buffer.baseAddress
            desc.hull_point_count = UInt32(points.count)
            desc.friction = friction
            desc.restitution = 0
            desc.motion = UJOLT_MOTION_KINEMATIC
            desc.gravity_factor = 0
            desc.layer = layer
            desc.position = (position.x, position.y, position.z)
            desc.rotation = (rotation.imag.x, rotation.imag.y, rotation.imag.z, rotation.real)
            desc.user_data = UInt64(Self.environmentEntity)
            let id = ujolt_world_add_body(worldHandle, &desc)
            guard id != UJOLT_INVALID_BODY else { return nil }
            return JoltKinematicBody(id: id)
        }
    }

    /// Where the body should be after the next step (applied as a kinematic
    /// move over that step, so it carries the implied velocity).
    func setKinematicTarget(_ body: JoltKinematicBody, position: simd_float3, rotation: simd_quatf) {
        // A non-finite target would poison the broadphase bounds next step.
        guard position.x.isFinite, position.y.isFinite, position.z.isFinite, rotation.vector.x.isFinite, rotation.vector.y.isFinite, rotation.vector.z.isFinite, rotation.vector.w.isFinite else { return }
        var p = (position.x, position.y, position.z)
        var r = (rotation.imag.x, rotation.imag.y, rotation.imag.z, rotation.real)
        withUnsafePointer(to: &p) { pp in
            withUnsafePointer(to: &r) { rp in
                pp.withMemoryRebound(to: Float.self, capacity: 3) { pf in
                    rp.withMemoryRebound(to: Float.self, capacity: 4) { rf in
                        ujolt_world_set_kinematic_target(worldHandle, body.id, pf, rf)
                    }
                }
            }
        }
    }

    /// Teleports the body (no implied velocity).
    func setKinematicTransform(_ body: JoltKinematicBody, position: simd_float3, rotation: simd_quatf) {
        guard position.x.isFinite, position.y.isFinite, position.z.isFinite, rotation.vector.x.isFinite, rotation.vector.y.isFinite, rotation.vector.z.isFinite, rotation.vector.w.isFinite else { return }
        var p = (position.x, position.y, position.z)
        var r = (rotation.imag.x, rotation.imag.y, rotation.imag.z, rotation.real)
        withUnsafePointer(to: &p) { pp in
            withUnsafePointer(to: &r) { rp in
                pp.withMemoryRebound(to: Float.self, capacity: 3) { pf in
                    rp.withMemoryRebound(to: Float.self, capacity: 4) { rf in
                        ujolt_world_set_transform(worldHandle, body.id, pf, rf)
                    }
                }
            }
        }
    }

    func removeKinematicBody(_ body: JoltKinematicBody) {
        ujolt_world_remove_body(worldHandle, body.id)
    }
}
