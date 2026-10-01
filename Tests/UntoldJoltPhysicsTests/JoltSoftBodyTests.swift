//
//  JoltSoftBodyTests.swift
//  UntoldJoltPhysics
//

import simd
import UntoldEngine
@testable import UntoldJoltPhysics
import XCTest

final class JoltSoftBodyTests: XCTestCase {
    private let step: Float = 1.0 / 60.0

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

    private func advance(_ backend: JoltPhysicsBackend, seconds: Float) {
        var elapsed: Float = 0
        while elapsed < seconds {
            backend.step(deltaTime: step)
            elapsed += step
        }
    }

    /// A horizontal square sheet of `n`×`n` particles, `spacing` apart, at
    /// height `y`, its four corners pinned; triangulated so it has a surface.
    private func sheet(n: Int, spacing: Float, y: Float) -> JoltSoftBodyDescriptor {
        var vertices: [SIMD3<Float>] = []
        var inverseMasses: [Float] = []
        let half = Float(n - 1) * spacing * 0.5
        for row in 0 ..< n {
            for column in 0 ..< n {
                vertices.append(SIMD3<Float>(Float(column) * spacing - half, y, Float(row) * spacing - half))
                let corner = (row == 0 || row == n - 1) && (column == 0 || column == n - 1)
                inverseMasses.append(corner ? 0 : 1.0 / 0.05)
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
        // Particles collide as spheres: their radius plus the ball's must
        // cover half the spacing, or the ball slips between them once the
        // sheet starts to oscillate.
        descriptor.vertexRadius = 0.03
        return descriptor
    }

    func testRopeHangsStraightDownFromItsPin() {
        let backend = makeBackend()
        // Two particles: the first pinned, the second held out sideways.
        var descriptor = JoltSoftBodyDescriptor(
            vertices: [SIMD3<Float>(0, 1.5, 0), SIMD3<Float>(0.3, 1.5, 0)],
            inverseMasses: [0, 1.0 / 0.01],
            edges: [SIMD2(0, 1)]
        )
        descriptor.compliance = 0
        descriptor.linearDamping = 2.0
        let rope = try! XCTUnwrap(backend.addSoftBody(descriptor))
        XCTAssertEqual(rope.vertexCount, 2)
        advance(backend, seconds: 3.0)
        var positions: [SIMD3<Float>] = []
        XCTAssertEqual(backend.readSoftBodyVertices(rope, into: &positions), 2)
        XCTAssertEqual(simd_length(positions[0] - SIMD3<Float>(0, 1.5, 0)), 0, accuracy: 1e-4, "The pin does not move")
        XCTAssertEqual(simd_length(positions[1] - positions[0]), 0.3, accuracy: 0.02, "The rope keeps its length")
        XCTAssertLessThan(positions[1].y, 1.5 - 0.27, "The free end hangs straight down")
    }

    func testSheetCatchesADroppedBall() throws {
        let backend = makeBackend()
        let sheet = try XCTUnwrap(backend.addSoftBody(sheet(n: 7, spacing: 0.1, y: 1.0)))
        advance(backend, seconds: 0.5)
        backend.didAddBody(entity: 7, descriptor: PhysicsBodyDescriptor(
            motionType: .dynamic,
            collider: PhysicsColliderDescriptor(shape: .sphere(radius: 0.06), friction: 0.4, restitution: 0.1),
            mass: 0.2,
            position: simd_float3(0, 1.6, 0)
        ))
        advance(backend, seconds: 2.0)
        let ball = try XCTUnwrap(backend.bodyState(for: 7))
        XCTAssertGreaterThan(ball.position.y, 0.5, "The sheet holds the ball up (the floor is at 0)")
        var positions: [SIMD3<Float>] = []
        backend.readSoftBodyVertices(sheet, into: &positions)
        let centre = positions[3 * 7 + 3]
        XCTAssertLessThan(centre.y, 0.98, "The sheet sags under the ball")
        XCTAssertEqual(positions[0].y, 1.0, accuracy: 1e-4, "A pinned corner stays put")
    }

    /// A cloth built from triangles (constraints derived, dihedral bends)
    /// hangs from two pinned corners; moving the pins with the setter
    /// carries the whole sheet along.
    func testClothFromFacesFollowsItsMovedPins() throws {
        let backend = makeBackend()
        let n = 8
        let spacing: Float = 0.05
        var vertices: [SIMD3<Float>] = []
        var inverseMasses: [Float] = []
        for row in 0 ..< n {
            for column in 0 ..< n {
                vertices.append(SIMD3<Float>(Float(column) * spacing, 1.5 - Float(row) * spacing, 0))
                inverseMasses.append(row == 0 && (column == 0 || column == n - 1) ? 0 : 1 / 0.01)
            }
        }
        var faces: [SIMD3<UInt32>] = []
        for row in 0 ..< n - 1 {
            for column in 0 ..< n - 1 {
                let a = UInt32(row * n + column), b = a + 1, c = a + UInt32(n), d = c + 1
                faces.append(SIMD3(a, c, b))
                faces.append(SIMD3(b, c, d))
            }
        }
        var descriptor = JoltSoftBodyDescriptor(
            vertices: vertices, inverseMasses: inverseMasses, faces: faces,
            compliance: 1e-6, shearCompliance: 1e-5, bendCompliance: 1e-3
        )
        descriptor.iterations = 8
        let body = try XCTUnwrap(backend.addSoftBody(descriptor))
        advance(backend, seconds: 1.0)

        var positions: [SIMD3<Float>] = []
        backend.readSoftBodyVertices(body, into: &positions)
        XCTAssertEqual(positions.count, n * n)
        XCTAssertTrue(positions.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite })
        let bottomBefore = positions[(n - 1) * n + n / 2]
        XCTAssertLessThan(bottomBefore.y, 1.5 - Float(n - 1) * spacing + 0.02, "hangs, does not float up")
        XCTAssertGreaterThan(bottomBefore.y, 1.5 - Float(n) * spacing - 0.1, "the constraints hold it together")

        // Move both pins 30 cm sideways: the sheet follows.
        let pins: [UInt32] = [0, UInt32(n - 1)]
        let targets = pins.map { vertices[Int($0)] + SIMD3<Float>(0.3, 0, 0) }
        for _ in 0 ..< 180 {
            XCTAssertEqual(backend.setSoftBodyVertices(body, indices: pins, worldPositions: targets), 2)
            backend.step(deltaTime: step)
        }
        backend.readSoftBodyVertices(body, into: &positions)
        XCTAssertEqual(positions[0].x, vertices[0].x + 0.3, accuracy: 1e-4)
        let bottomAfter = positions[(n - 1) * n + n / 2]
        XCTAssertGreaterThan(bottomAfter.x, bottomBefore.x + 0.15, "the free bottom came along (it swings for a while)")

        // An index out of range is ignored, not applied.
        XCTAssertEqual(backend.setSoftBodyVertices(body, indices: [UInt32(n * n)], worldPositions: [.zero]), 0)
        backend.removeSoftBody(body)
    }

    /// A hanging cloth draped over a kinematic capsule that is moved
    /// through it stays finite and ends up pushed aside, not passed through.
    func testClothCollidesWithAMovedKinematicCapsule() throws {
        let backend = makeBackend()
        let n = 10
        let spacing: Float = 0.04
        var vertices: [SIMD3<Float>] = []
        var inverseMasses: [Float] = []
        for row in 0 ..< n {
            for column in 0 ..< n {
                vertices.append(SIMD3<Float>(Float(column) * spacing - 0.18, 1.2 - Float(row) * spacing, 0))
                inverseMasses.append(row == 0 ? 0 : 1 / 0.01)
            }
        }
        var faces: [SIMD3<UInt32>] = []
        for row in 0 ..< n - 1 {
            for column in 0 ..< n - 1 {
                let a = UInt32(row * n + column), b = a + 1, c = a + UInt32(n), d = c + 1
                faces.append(SIMD3(a, c, b))
                faces.append(SIMD3(b, c, d))
            }
        }
        var descriptor = JoltSoftBodyDescriptor(
            vertices: vertices, inverseMasses: inverseMasses, faces: faces,
            compliance: 1e-6, shearCompliance: 1e-5, bendCompliance: 1e-3
        )
        descriptor.vertexRadius = 0.01
        let cloth = try XCTUnwrap(backend.addSoftBody(descriptor))
        // A horizontal capsule behind the sheet, moved forward through its plane.
        let sideways = simd_quatf(angle: .pi / 2, axis: SIMD3<Float>(0, 0, 1))
        let capsule = try XCTUnwrap(backend.addKinematicCapsule(radius: 0.05, height: 0.5, position: SIMD3<Float>(0, 1.0, -0.2), rotation: sideways))
        advance(backend, seconds: 0.5)
        for frame in 0 ..< 90 {
            let z = -0.2 + 0.4 * Float(frame) / 90
            backend.setKinematicTarget(capsule, position: SIMD3<Float>(0, 1.0, z), rotation: sideways)
            backend.step(deltaTime: step)
        }
        var positions: [SIMD3<Float>] = []
        backend.readSoftBodyVertices(cloth, into: &positions)
        XCTAssertTrue(positions.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite })
        // The row at the capsule's height was carried forward with it.
        let rowAtCapsule = 5
        let pushed = positions[rowAtCapsule * n + n / 2].z
        XCTAssertGreaterThan(pushed, 0.12, "cloth pushed in front of the capsule, at z \(pushed)")
        backend.removeKinematicBody(capsule)
        backend.removeSoftBody(cloth)
    }

    func testClothCollidesWithAMovedKinematicConvexHull() throws {
        let backend = makeBackend()
        let n = 10
        let spacing: Float = 0.04
        var vertices: [SIMD3<Float>] = []
        var inverseMasses: [Float] = []
        for row in 0 ..< n {
            for column in 0 ..< n {
                vertices.append(SIMD3<Float>(Float(column) * spacing - 0.18, 1.2 - Float(row) * spacing, 0))
                inverseMasses.append(row == 0 ? 0 : 1 / 0.01)
            }
        }
        var faces: [SIMD3<UInt32>] = []
        for row in 0 ..< n - 1 {
            for column in 0 ..< n - 1 {
                let a = UInt32(row * n + column), b = a + 1, c = a + UInt32(n), d = c + 1
                faces.append(SIMD3(a, c, b))
                faces.append(SIMD3(b, c, d))
            }
        }
        var descriptor = JoltSoftBodyDescriptor(
            vertices: vertices, inverseMasses: inverseMasses, faces: faces,
            compliance: 1e-6, shearCompliance: 1e-5, bendCompliance: 1e-3
        )
        descriptor.vertexRadius = 0.01
        let cloth = try XCTUnwrap(backend.addSoftBody(descriptor))
        // A wedge (a box with a squashed top) behind the sheet, moved forward through its plane.
        var points: [simd_float3] = []
        for x: Float in [-0.25, 0.25] {
            for z: Float in [-0.05, 0.05] {
                points.append(simd_float3(x, -0.05, z))
                points.append(simd_float3(x * 0.6, 0.05, z * 0.6))
            }
        }
        XCTAssertNil(backend.addKinematicConvexHull(points: [simd_float3(0, 0, 0), simd_float3(1, 0, 0), simd_float3(0, 1, 0)], position: .zero, rotation: simd_quatf(angle: 0, axis: simd_float3(0, 1, 0))), "three points are no hull")
        let hull = try XCTUnwrap(backend.addKinematicConvexHull(points: points, position: SIMD3<Float>(0, 1.0, -0.2), rotation: simd_quatf(angle: 0, axis: simd_float3(0, 1, 0))))
        advance(backend, seconds: 0.5)
        for frame in 0 ..< 90 {
            let z = -0.2 + 0.4 * Float(frame) / 90
            backend.setKinematicTarget(hull, position: SIMD3<Float>(0, 1.0, z), rotation: simd_quatf(angle: 0, axis: simd_float3(0, 1, 0)))
            backend.step(deltaTime: step)
        }
        var positions: [SIMD3<Float>] = []
        backend.readSoftBodyVertices(cloth, into: &positions)
        XCTAssertTrue(positions.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite })
        let rowAtHull = 5
        let pushed = positions[rowAtHull * n + n / 2].z
        XCTAssertGreaterThan(pushed, 0.12, "cloth pushed in front of the hull, at z \(pushed)")
        backend.removeKinematicBody(hull)
        backend.removeSoftBody(cloth)
    }

    func testRemovedSoftBodyStopsExisting() throws {
        let backend = makeBackend()
        let body = try XCTUnwrap(backend.addSoftBody(sheet(n: 3, spacing: 0.1, y: 1.0)))
        advance(backend, seconds: 0.2)
        backend.removeSoftBody(body)
        advance(backend, seconds: 0.2)
        var positions: [SIMD3<Float>] = []
        XCTAssertEqual(backend.readSoftBodyVertices(body, into: &positions), 0)
        XCTAssertNil(backend.addSoftBody(JoltSoftBodyDescriptor(vertices: [.zero, .zero], inverseMasses: [0, 1], edges: [SIMD2(0, 5)])), "A bad edge index is refused")
    }
}
