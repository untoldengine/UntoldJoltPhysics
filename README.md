# UntoldJoltPhysics

[Jolt Physics](https://github.com/jrouwe/JoltPhysics) as a physics backend
plugin for the [Untold Engine](https://github.com/untoldengine/UntoldEngine).

The engine ships a physics-plugin seam (`PhysicsBackend`, `PhysicsBackendPlugin`,
`PhysicsBackendRegistry`, the `PhysicsCoordinator` engine extension, `PhysicsEvents`
and `PhysicsQuery`). This package installs Jolt behind that seam: rigid bodies
described with the engine-owned `RigidBodyComponent`/`ColliderComponent` are
simulated by Jolt, transforms flow back into the ECS every fixed substep, and
Jolt's contacts, sensors and sleep state arrive as engine physics events. No
engine changes and no binaries: Jolt comes as source from the
[untoldengine/JoltPhysics](https://github.com/untoldengine/JoltPhysics) fork — the
upstream tree plus a `Package.swift` — and SwiftPM compiles it as a C++17
target for macOS, iOS and visionOS (device and simulator).

## Layout

Follows the engine's [plugin authoring guidelines](https://github.com/untoldengine/UntoldEngine/blob/develop/docs/Extensions/PluginAuthoringGuidelines.md):

| Path | What it is |
|---|---|
| `Package.swift` | Depends on `JoltPhysics` from the fork at an exact tag (`5.6.0-spm.1` = Jolt v5.6.0). To update Jolt: branch the fork from the new upstream tag, keep its `Package.swift`, tag `<version>-spm.1`, bump the pin here. |
| `Sources/CJoltBridge/` | C ABI shim over Jolt (`CJoltBridge.h`): opaque world handle, plain structs, explicit create/destroy — world, bodies, kinematic targets, step, transform read-back, buffered contact/activation events, ray cast. |
| `Sources/UntoldJoltPhysics/` | The plugin: `JoltPhysicsBackend` (conforms to `PhysicsBackend`), `JoltPhysicsPlugin` (the manifest) and `registerJoltPhysics()`. |
| `Tests/` | Backend tests driven through the engine protocol, plus manifest/registration tests. |

## Install

```swift
// Package.swift
.package(url: "https://github.com/untoldengine/UntoldJoltPhysics.git", from: "0.1.0"),
// target dependencies
.product(name: "UntoldJoltPhysics", package: "UntoldJoltPhysics"),
```

## Register

```swift
import UntoldJoltPhysics

// Before the renderer is created: the registry locks on the first substep.
let jolt = registerJoltPhysics()
```

`registerJoltPhysics(settings:)` installs the plugin through
`PhysicsBackendRegistry` and returns the live backend (or, when the registry is
already locked because an immersive space was reopened, the backend that is
already installed). Then give entities the engine's physics components as
usual — `RigidBodyComponent` (motion type, mass, layer) and `ColliderComponent`
(sphere, box, capsule, cylinder, convex hull; `isTrigger` for sensor volumes).
Contacts, trigger enter/exit and activation changes are published through
`PhysicsEvents`; `PhysicsQuery.raycast` is answered by Jolt (the backend
declares the `.raycast` capability).

`JoltWorldSettings` tunes the world: body/pair/contact capacities, worker
threads (`0` for a deterministic single-threaded step), persisted-contact
reporting, and read-back capacity.

## Platforms and compatibility

macOS 14+, iOS 17+, visionOS 2+ (the engine's minimums). Builds for the
visionOS simulator too.

```text
Plugin Version | Untold Engine Version
---------------|-------------------------------------------
0.1.x          | develop, physics plugin API version 1
               | (seam complete since 0.18.0; verified at 32561b4a and 0.19.x)
```

The manifest requires the exact `PhysicsBackendAPIVersion.current` it was
built against, as the engine's validator demands.

### Semantics worth knowing

- **Layers.** The engine's `layer` (0…31) maps to a Jolt object layer; the
  world configuration's `collisionLayerMatrix` (`masks[i]` = layers that layer
  `i` collides with, symmetrised) drives Jolt's pair filter. Static bodies never
  collide with each other. Per-body `collisionMask` is not consulted yet.
- **Contact events** follow the engine's reference backend: `entityA` is the
  dynamic body (the second body when both or neither are) and the normal points
  out of `entityB` into `entityA`. The impulse in `.began` events is an estimate
  (closing speed along the normal times the reduced mass); Jolt does not expose
  the solved impulse in its callback.
- **Triggers** are kinematic, never-sleeping Jolt sensors even when declared
  static, so occupancy is reported regardless of whether the occupant sleeps;
  they see dynamic bodies only.
- **Kinematic bodies** are moved with `MoveKinematic` over the substep, so they
  carry the velocity implied by the engine's transform writes and impart it on
  what they hit. `maxKinematicStep` turns a far jump into a teleport and
  `maxKinematicSpeed` caps what a near jump can carry, so tracking glitches
  never become impacts.
- **Continuous collision** (`continuousCollision`, on by default) gives dynamic
  bodies Jolt's LinearCast motion quality: a fast ball cannot pass through a
  thin backboard or a 4 cm surface slab.
- **Read-back** returns the dynamic bodies that are active plus those that fell
  asleep during the step; static and kinematic bodies are never read back.
- **Removed bodies** keep their identity until the next event drain, so a
  trigger exit for a body deleted while inside the volume is still reported.
- **Resting contacts** are not impacts: a non-trigger `.began` is only reported
  when the closing speed exceeds `minContactSpeed`, and `.ended` only follows a
  reported `.began`. Triggers see dynamic bodies only; rays pass through them.
- **Environment boxes** (`setEnvironmentBoxes`) are entity-less static slabs
  for detected real-world surfaces; contacts and ray hits on them carry
  `JoltPhysicsBackend.environmentEntity` (the engine's null entity). Re-sent
  boxes that did not change keep their Jolt bodies, so resting balls stay put.
- **Restitution/friction** combine with Jolt's defaults (max / geometric mean).
- **Soft bodies** (`JoltPhysicsBackend.addSoftBody` / `readSoftBodyVertices` /
  `removeSoftBody`, a side channel like `resetBody`): particles joined by
  distance constraints, solved by Jolt's own XPBD, colliding both ways with
  the rigid bodies. Pinned particles (inverse mass 0) hold the body in place;
  faces are optional and give it a surface for continuous collision. Rigid
  bodies collide with the particles as spheres of `vertexRadius`, so that
  radius plus the other body's must cover half the particle spacing or the
  body slips through the gaps. The engine seam has no soft-body vocabulary:
  a game reads the vertex positions back each frame and drives a mesh.
- **Character controller** (`JoltPhysicsBackend.addCharacter` /
  `removeCharacter`, then `JoltCharacter.move` / `teleport` / `position` /
  `contacts`): Jolt's `CharacterVirtual`, a capsule or cylinder standing on
  its base. The world step never moves it; the game moves it every frame with
  the velocity its animation produced, the controller collides and slides that
  motion against the environment boxes and the rigid bodies, and the game reads
  the corrected position back. Gravity is never added to its velocity (pass one
  to `move` only to press on what it stands on), so a game that owns the
  character's height keeps it. It pushes dynamic bodies (up to `maxStrength`)
  and, unless `pushedByDynamicBodies` is set, is never shoved by them; kinematic
  bodies push it either way. An inner kinematic body (`innerBody`, 90% of the
  shape) makes dynamic bodies bounce off the character and rays hit it; it
  carries the character's `entity`, is never read back and goes with the
  character. Every character call is frame-thread only, between steps.

## Tests

```bash
swift test
```

The backend suite drives the plugin through the engine's protocol the way the
coordinator does: bounce and rest, static boxes, triggers (including exit on
removal), a kinematic swat, body removal and replacement, raycasts with
exclusions and layer masks, the layer matrix, capsule/cylinder/convex-hull
shapes, collider offsets, gravity, and soft bodies (a hanging rope, a sheet
catching a ball, removal). The character suite walks a controller into a wall
and along it, over a low box, into a ball it pushes and a ball that bounces
off it, hits it with a ray, teleports it and removes it. The plugin suite
covers the manifest, install/uninstall, replacement and the registration
helper.

## Roadmap

Staged as agreed with the engine maintainers: prototype (this) → collision
events (done) → character controller (done) → mesh colliders.

## License

Licensed under the [MIT License](LICENSE), same as Jolt Physics itself; see
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for Jolt's own notice.
