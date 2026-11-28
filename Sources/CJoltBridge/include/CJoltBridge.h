//
//  CJoltBridge.h
//  UntoldJoltPhysics
//
//  C API over Jolt Physics for the Untold Engine physics backend plugin.
//  Deliberately narrow: it mirrors the engine's PhysicsBackend protocol
//  (bodies in, kinematic targets in, step, transforms out, buffered events
//  out, one raycast) so the Swift side stays a thin adapter, plus the plugin's
//  own extras (soft bodies, a character controller, ragdolls). All functions except
//  the event drains must be called from one thread (the engine's frame
//  thread); Jolt's own worker threads never call back into Swift.
//

#ifndef CJOLTBRIDGE_H
#define CJOLTBRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ujolt_world ujolt_world;

/// Jolt BodyID (index + sequence number). Never reused while the body lives.
typedef uint32_t ujolt_body_id;
#define UJOLT_INVALID_BODY 0xFFFFFFFFu
/// user_data of bodies that belong to no entity (environment geometry) and of
/// "no body" answers. Equals the engine's EntityID.invalid.
#define UJOLT_NO_ENTITY 0xFFFFFFFFFFFFFFFFull

typedef enum ujolt_motion_type {
    UJOLT_MOTION_STATIC = 0,
    UJOLT_MOTION_KINEMATIC = 1,
    UJOLT_MOTION_DYNAMIC = 2
} ujolt_motion_type;

typedef enum ujolt_shape_type {
    UJOLT_SHAPE_SPHERE = 0,
    UJOLT_SHAPE_BOX = 1,
    UJOLT_SHAPE_CAPSULE = 2,
    UJOLT_SHAPE_CYLINDER = 3,
    UJOLT_SHAPE_CONVEX_HULL = 4
} ujolt_shape_type;

typedef struct ujolt_world_desc {
    uint32_t max_bodies;              /* 0 -> 4096 */
    uint32_t max_body_pairs;          /* 0 -> 4096 */
    uint32_t max_contact_constraints; /* 0 -> 2048 */
    uint32_t temp_allocator_bytes;    /* 0 -> 16 MiB */
    int32_t worker_threads;           /* <0 -> cores-1; 0 -> single-threaded (deterministic) */
    int32_t report_persisted_contacts;/* 0/1: also emit UJOLT_CONTACT_PERSISTED every step */
    float gravity[3];
    /* A kinematic target farther than this (metres) from the body's current
       position teleports it instead of moving it: a tracked hand re-appearing
       100 m away must not launch what it lands on. <= 0 -> unlimited. */
    float max_kinematic_step;
    /* Ceiling (m/s) on the velocity a kinematic move may carry: a target
       within max_kinematic_step but arriving fast (a tracking glitch, a
       stalled frame) approaches at this speed over a few substeps instead
       of hitting at the implied one. <= 0 -> unlimited. */
    float max_kinematic_speed;
    /* 0/1: dynamic bodies use Jolt's LinearCast motion quality (continuous
       collision), so a fast body cannot pass through thin geometry. */
    int32_t continuous_collision;
    /* Non-sensor contacts whose closing speed (m/s) is below this are not
       reported as ADDED (resting contacts re-established after a sleep or a
       geometry rebuild are not impacts). REMOVED follows only a reported
       ADDED. <= 0 -> report everything. */
    float min_contact_speed;
} ujolt_world_desc;

typedef struct ujolt_body_desc {
    ujolt_shape_type shape;
    float radius;                 /* sphere, capsule, cylinder */
    float half_height;            /* capsule (cylindrical part), cylinder */
    float half_extents[3];        /* box */
    const float *hull_points;     /* convex hull: xyz triples */
    uint32_t hull_point_count;
    float local_offset[3];        /* shape offset from the body origin */
    float friction;
    float restitution;
    int32_t is_sensor;            /* trigger volume: reports contacts, no response */
    ujolt_motion_type motion;
    float mass;                   /* dynamic bodies; <= 0 -> from shape volume */
    uint32_t layer;               /* engine collision layer 0..31 */
    float gravity_factor;
    float position[3];
    float rotation[4];            /* x, y, z, w */
    float linear_velocity[3];
    float angular_velocity[3];
    uint64_t user_data;           /* the engine entity id */
} ujolt_body_desc;

/// A vertex skinned to the joints of an animated skeleton: Jolt's skinned
/// constraint. Every step the game feeds the joints' world matrices
/// (ujolt_world_skin_soft_body); the bridge skins the vertex's rest
/// position with them and the solver keeps the simulated particle within
/// max_distance of that skinned position and outside a back-stop sphere
/// behind it (the body the cloth is worn over), or exactly on it when
/// max_distance is 0. "Behind" is against the skin's normal, which Jolt
/// takes from the faces around the vertex whose three vertices are all
/// skinned, wound counter-clockwise seen from outside the cloth; a
/// back-stopped vertex with no such face is refused.
typedef struct ujolt_soft_body_skin_vertex {
    uint32_t vertex;             /* index into the descriptor's vertices */
    uint32_t joints[4];          /* up to four joints, read up to the first zero weight */
    float weights[4];            /* normalised by the bridge */
    float max_distance;          /* 0 = hard skinned: the bridge also sets its inverse mass to 0 */
    float backstop_distance;     /* behind the skinned vertex, against its normal; disabled when
                                    >= max_distance x the skin's max-distance multiplier */
    float backstop_radius;       /* 40 approximates a plane */
} ujolt_soft_body_skin_vertex;

/// A soft body: particles joined by distance constraints (Jolt's own XPBD),
/// optional triangles for surface collision. Vertices are relative to
/// `position`; the body never moves as a whole (`position` is its frame),
/// pinned vertices (inverse mass 0) hold it in place.
typedef struct ujolt_soft_body_desc {
    const float *vertices;         /* xyz triples */
    const float *inv_masses;       /* per vertex; 0 = pinned. NULL -> all 1 */
    uint32_t vertex_count;
    const uint32_t *edges;         /* index pairs */
    const float *edge_compliances; /* per edge, or NULL -> compliance */
    float compliance;              /* inverse stiffness, m/N; 0 = rigid */
    uint32_t edge_count;
    const uint32_t *faces;         /* index triples; optional. Counter-clockwise seen from
                                      outside: a skin's back-stop lies behind their normal */
    uint32_t face_count;
    float position[3];
    uint32_t layer;                /* engine collision layer 0..31 */
    uint32_t iterations;           /* solver iterations per step; 0 -> Jolt's default */
    float linear_damping;
    float vertex_radius;           /* collision radius of a vertex */
    float friction;
    float restitution;
    float gravity_factor;
    uint64_t user_data;
    /* Cloth from triangles: when non-zero, the distance, shear and bend
       constraints are built from `faces` (Jolt's CreateConstraints with
       dihedral bends) using `compliance`, `shear_compliance` and
       `bend_compliance`, and `edges` is ignored. */
    int32_t constraints_from_faces;
    float shear_compliance;
    float bend_compliance;
    /* Ceiling on a vertex's speed (m/s); <= 0 -> Jolt's default. Bounds the
       energy a resolved overlap or a yanked pin can put into the body. */
    float max_linear_velocity;
    /* Bend constraints derived from the faces: 0 = dihedral (angle between
       neighbouring triangles, stiffer, less stable when pins move fast),
       1 = distance (across the shared edge, robust). */
    int32_t bend_type;
    /* Skinning (Jolt's skinned constraints). inv_bind holds joint_count 4x4
       matrices, 16 floats each, column-major: each takes a vertex in the
       descriptor's vertex frame (relative to position) to its joint's
       space. skinned_count > 0 requires face_count > 0 (the back-stop
       normals come from the faces) and joint_count > 0. The body is created
       skinned to its bind pose, so a step before its first feed is sound. */
    const float *inv_bind;
    uint32_t joint_count;
    const ujolt_soft_body_skin_vertex *skinned;
    uint32_t skinned_count;
    /* 0/1: whether Jolt may put the body to sleep at rest. Jolt never wakes
       a body for its skin, so cloth on a live character passes 0 rather
       than rely on the feed waking it every frame. */
    int32_t allow_sleeping;
    /* 0/1: ragdoll parts push this body and it never pushes them (their
       inverse mass is scaled to 0 in the contact, so a moving dynamic part
       still drags it by friction; a kinematic part only pushes); every
       other body collides both ways. The one gap is Jolt's continuous
       collision: a dynamic part moving more than three quarters of its
       inner radius in a sub-step is still placed at the time of impact
       with the cloth for that sub-step (no impulse, but the rest of that
       sub-step's travel is lost); raise collision_steps or turn continuous
       collision off if it shows. */
    int32_t one_way_against_ragdolls;
} ujolt_soft_body_desc;

typedef enum ujolt_contact_phase {
    UJOLT_CONTACT_ADDED = 0,
    UJOLT_CONTACT_PERSISTED = 1,
    UJOLT_CONTACT_REMOVED = 2
} ujolt_contact_phase;

typedef struct ujolt_contact_event {
    ujolt_contact_phase phase;
    uint64_t user_a;
    uint64_t user_b;
    int32_t sensor_a;
    int32_t sensor_b;
    /* A is the dynamic body (body 2 when both or neither are); the normal
       points out of B into A. Zero position/normal/impulse for REMOVED. */
    float position[3];  /* first manifold point */
    float normal[3];
    float impulse;      /* estimated normal impulse in N*s */
} ujolt_contact_event;

typedef struct ujolt_activation_event {
    uint64_t user_data;
    int32_t is_active;
} ujolt_activation_event;

typedef struct ujolt_ray_hit {
    uint64_t user_data;
    float position[3];
    float normal[3];
    float distance;
} ujolt_ray_hit;

/* World lifecycle */
ujolt_world *ujolt_world_create(const ujolt_world_desc *desc);
void ujolt_world_destroy(ujolt_world *world);
void ujolt_world_set_gravity(ujolt_world *world, const float gravity[3]);
/// layer_masks[i] = bitmask of the layers that layer i collides with. NULL or
/// count 0 restores "everything collides".
void ujolt_world_set_layer_matrix(ujolt_world *world, const uint32_t *layer_masks, uint32_t count);

/* Bodies */
ujolt_body_id ujolt_world_add_body(ujolt_world *world, const ujolt_body_desc *desc);
void ujolt_world_remove_body(ujolt_world *world, ujolt_body_id body);

/* Soft bodies (removed with ujolt_world_remove_body) */
ujolt_body_id ujolt_world_add_soft_body(ujolt_world *world, const ujolt_soft_body_desc *desc);
uint32_t ujolt_world_soft_body_vertex_count(ujolt_world *world, ujolt_body_id body);
/// World-space vertex positions, xyz triples. Returns the count written
/// (capped at capacity).
uint32_t ujolt_world_read_soft_body_vertices(ujolt_world *world, ujolt_body_id body, float *positions, uint32_t capacity);
/// Moves `count` vertices (by index) to world-space positions (xyz triples)
/// and clears their velocity; meant for pinned vertices that ride on
/// something animated. Returns the count applied.
uint32_t ujolt_world_set_soft_body_vertices(ujolt_world *world, ujolt_body_id body, const uint32_t *indices, const float *positions, uint32_t count);
/// Feeds the joints for this step: joint_count world-space 4x4 matrices,
/// 16 floats each, column-major; the bridge makes them relative to the
/// body's centre of mass. hard_skin_all places every skinned vertex on its
/// skinned position and zeroes its velocity (a teleport); a body's first
/// feed is a hard skin whatever it says, since Jolt blends the skinned
/// positions from the previous feed's and there is none yet (the body is
/// created skinned to its bind pose, so a step before that feed is sound).
/// Frame thread, between steps only, never from inside a step. Returns 0
/// when the body is not a skinned soft body, a matrix is not finite or
/// joint_count does not cover the skin's joints (nothing is applied then),
/// 1 otherwise.
int32_t ujolt_world_skin_soft_body(ujolt_world *world, ujolt_body_id body, const float *joint_world_matrices, uint32_t joint_count, int32_t hard_skin_all);
/// Jolt's SetSkinnedMaxDistanceMultiplier (scales every skinned vertex's
/// max_distance; 0 hard skins everything, and a multiplier below a vertex's
/// backstop_distance / max_distance switches that back-stop off) and
/// SetEnableSkinConstraints (off, only the hard-skinned vertices still
/// follow the skin). A non-finite multiplier applies nothing.
void ujolt_world_set_soft_body_skin_params(ujolt_world *world, ujolt_body_id body, float max_distance_multiplier, int32_t enable_skin_constraints);
/// World-space vertex positions (xyz triples) and, when normals is not
/// NULL, unit area-weighted vertex normals from the body's faces (a vertex
/// on no face gets +Y). Returns the count written (capped at capacity).
uint32_t ujolt_world_read_soft_body_vertices_and_normals(ujolt_world *world, ujolt_body_id body, float *positions, float *normals, uint32_t capacity);
uint32_t ujolt_world_body_count(const ujolt_world *world);
uint64_t ujolt_world_get_user_data(const ujolt_world *world, ujolt_body_id body);

/// Buffered until the next step, which applies it as a kinematic move over
/// that step's dt (so the body carries the implied velocity).
void ujolt_world_set_kinematic_target(ujolt_world *world, ujolt_body_id body, const float position[3], const float rotation[4]);
/// Immediate teleport (any motion type).
void ujolt_world_set_transform(ujolt_world *world, ujolt_body_id body, const float position[3], const float rotation[4]);
void ujolt_world_set_velocity(ujolt_world *world, ujolt_body_id body, const float linear[3], const float angular[3]);
void ujolt_world_get_transform(const ujolt_world *world, ujolt_body_id body, float position[3], float rotation[4]);
void ujolt_world_get_velocity(const ujolt_world *world, ujolt_body_id body, float linear[3], float angular[3]);
int32_t ujolt_world_body_is_active(const ujolt_world *world, ujolt_body_id body);
/// Puts a body to sleep (0) or wakes it (1): Jolt's DeactivateBody and
/// ActivateBody, for a soft body as much as a rigid one. Asleep, a body is
/// left out of the step — no integration, no constraints, no collision of
/// its own, and for a soft body no solve and no skinning — until something
/// wakes it: this call, a skin feed or a skin-parameter change (the bridge
/// wakes the body for those, as a sleeping body would not follow what it
/// was just given), or an awake rigid body reaching its bounds on a layer
/// the two share: Jolt pairs an active rigid body with a sleeping soft body
/// and activates the soft body before it leaves the pair to the soft-body
/// update (PhysicsSystem::ProcessBodyPair). A body to be kept asleep sits on
/// a layer nothing awake meets (ujolt_world_set_body_layer). Frame thread,
/// between steps.
void ujolt_world_set_body_active(ujolt_world *world, ujolt_body_id body, int32_t active);
/// Moves a body to another engine layer (0..31), keeping whether it counts
/// as moving; the broad phase follows at once and the layer matrix decides
/// what meets it from the next step. Frame thread, between steps.
void ujolt_world_set_body_layer(ujolt_world *world, ujolt_body_id body, uint32_t layer);

/// Wakes every body overlapping the world-space box [min, max] — call before
/// removing static geometry a sleeping body may be resting on.
void ujolt_world_activate_in_box(ujolt_world *world, const float min[3], const float max[3]);

/* Simulation */
void ujolt_world_step(ujolt_world *world, float dt, int32_t collision_steps);

/// Dynamic bodies whose transform may have changed in the last step: the
/// active ones plus those that fell asleep during it. Returns the count
/// written (capped at capacity).
uint32_t ujolt_world_changed_bodies(ujolt_world *world, ujolt_body_id *ids, uint32_t capacity);

/* Events buffered during the step (thread-safe against Jolt's workers) */
uint32_t ujolt_world_drain_contacts(ujolt_world *world, ujolt_contact_event *out, uint32_t capacity, uint32_t *dropped);
uint32_t ujolt_world_drain_activations(ujolt_world *world, ujolt_activation_event *out, uint32_t capacity, uint32_t *dropped);

/* Queries */
/// Closest hit along origin + direction * t, t in [0, max_distance]. Returns 1 on hit.
int32_t ujolt_world_cast_ray(const ujolt_world *world, const float origin[3], const float direction[3], float max_distance,
                             uint32_t layer_mask, const uint64_t *excluded_user_data, uint32_t excluded_count,
                             ujolt_ray_hit *out_hit);

/* Character controller (Jolt's CharacterVirtual): a shape the game moves
   with a velocity every frame. It collides-and-slides against the static
   environment and the rigid bodies, pushes dynamic ones, and reports the
   corrected position. The world step never moves it: the game calls
   ujolt_character_move between steps, on the frame thread. */
typedef struct ujolt_character ujolt_character;

typedef enum ujolt_character_shape {
    UJOLT_CHARACTER_CAPSULE = 0,
    UJOLT_CHARACTER_CYLINDER = 1
} ujolt_character_shape;

typedef enum ujolt_ground_state {
    UJOLT_GROUND_ON_GROUND = 0,
    UJOLT_GROUND_ON_STEEP_GROUND = 1,
    UJOLT_GROUND_NOT_SUPPORTED = 2,
    UJOLT_GROUND_IN_AIR = 3
} ujolt_ground_state;

typedef struct ujolt_character_desc {
    ujolt_character_shape shape;
    float radius;
    float height;                      /* total height, base (feet) to top; a capsule needs >= 2 * radius */
    float position[3];                 /* the base of the shape (the feet) */
    float rotation[4];                 /* x, y, z, w */
    uint32_t layer;                    /* engine collision layer 0..31 */
    uint64_t user_data;                /* the engine entity id; contacts and ray hits report it */
    float mass;                        /* kg, presses down on what the character stands on; 0 never does; < 0 -> 70 */
    float max_strength;                /* N, the most force applied to a pushed dynamic body; 0 never pushes; < 0 -> 100 */
    float padding;                     /* <= 0 -> 0.02 m; the distance kept from every surface */
    float predictive_contact_distance; /* <= 0 -> 0.1 m; 0 would make the character stick */
    float max_slope_degrees;           /* contacts steeper than this are walls; 0 makes every contact a wall; < 0 -> 50 */
    float penetration_recovery_speed;  /* <= 0 -> 1; fraction of a penetration resolved per move */
    int32_t inner_body;                /* 0/1: a kinematic body inside the shape, so dynamic bodies
                                          bounce off the character and rays hit it (registered with
                                          user_data; never read back, removed with the character) */
    float inner_body_fraction;         /* <= 0 -> 0.9 of the outer shape */
    int32_t ignores_dynamic_bodies;    /* 0/1: the character's own collision (blocking, pushing) does not
                                          see dynamic bodies at all; they meet only its inner body, whose
                                          contacts the world reports. For a character whose hits a game
                                          wants to hear about. */
    int32_t pushed_by_dynamic_bodies;  /* 0/1: whether dynamic bodies may shove the character. Off, a
                                          ball resting against it or hitting it never moves it, while
                                          the character still pushes the ball; kinematic bodies (a
                                          tracked hand) push it either way */
    float stick_to_floor_step_down;    /* metres the character may snap down to stay on a floor; 0 -> off */
    float walk_stairs_step_up;         /* metres the character may step up; 0 -> off */
} ujolt_character_desc;

/// A contact of the character with a body. Trigger volumes are never listed:
/// the character passes through them.
typedef struct ujolt_character_contact {
    uint64_t user_data;   /* the other body's entity (UJOLT_NO_ENTITY for environment geometry) */
    float position[3];
    float normal[3];      /* points towards the character */
    float distance;       /* <= 0 touching or penetrating, > 0 a predicted contact ahead */
    int32_t is_dynamic;
    int32_t had_collision;/* 1 when the last move actually collided with it */
} ujolt_character_contact;

ujolt_character *ujolt_world_add_character(ujolt_world *world, const ujolt_character_desc *desc);
void ujolt_world_remove_character(ujolt_world *world, ujolt_character *character);
/// Moves by velocity * dt with collide-and-slide. Gravity only presses on
/// whatever the character stands on (pass zeros when the game owns the
/// height); it is never added to the character's velocity. Frame thread,
/// between steps.
void ujolt_character_move(ujolt_character *character, const float velocity[3], float dt, const float gravity[3]);
/// Teleports the base; the contacts are recomputed.
void ujolt_character_set_position(ujolt_character *character, const float position[3]);
void ujolt_character_set_rotation(ujolt_character *character, const float rotation[4]);
void ujolt_character_get_position(const ujolt_character *character, float position[3]);
/// The velocity the last move actually produced — the displacement over its
/// dt, after sliding and stopping — not the one asked for. Zero before any move.
void ujolt_character_get_velocity(const ujolt_character *character, float velocity[3]);
int32_t ujolt_character_ground_state(const ujolt_character *character);
/// The contacts the last move found. Writes up to capacity of them and
/// returns the TOTAL count, so a caller can grow its buffer and ask again.
uint32_t ujolt_character_contacts(const ujolt_character *character, ujolt_character_contact *out, uint32_t capacity);
/// The inner body, or UJOLT_INVALID_BODY when the character has none.
ujolt_body_id ujolt_character_inner_body(const ujolt_character *character);

/* Ragdoll (Jolt's Ragdoll over SwingTwist constraints): a tree of rigid
   parts, one per skeleton joint, each joined to its parent by a swing-twist
   joint with a motor. A part is kinematic (it follows the pose the game
   gives) or dynamic (simulated; its motor, when on, pulls it toward the
   pose the game drives), so a knockdown is every part dynamic with the
   motors off and some joint friction, and a hit reaction is the hit
   subtree dynamic with motors driving it toward the live animation while
   the rest stays kinematic. Every part reports the ragdoll's user_data in
   contacts and rays; none is ever listed by ujolt_world_changed_bodies nor
   removable with ujolt_world_remove_body. All calls are frame thread,
   between steps. */
typedef struct ujolt_ragdoll ujolt_ragdoll;

typedef enum ujolt_motor_mode {
    UJOLT_MOTOR_OFF = 0,
    UJOLT_MOTOR_POSITION = 1
} ujolt_motor_mode;

typedef struct ujolt_ragdoll_part {
    const char *name;
    int32_t parent;                 /* index into the parts array; -1 for the root part; parents come before children */
    ujolt_shape_type shape;         /* UJOLT_SHAPE_CAPSULE, _SPHERE or _BOX (a cylinder is taken too; never a hull) */
    float radius;                   /* sphere, capsule */
    float half_height;              /* capsule (cylindrical part) */
    float half_extents[3];          /* box */
    float shape_offset[3];          /* the shape's centre in the part's frame (the body origin is the joint pivot) */
    float shape_rotation[4];        /* the shape's rotation in the part's frame, x y z w (a capsule/cylinder axis is its local Y) */
    float mass;                     /* kg; <= 0 -> from the shape's volume. Jolt's stabilisation then clamps every
                                       parent/child mass ratio to 0.8..1.2, keeping the total */
    float position[3];              /* the neutral pose, world: the joint pivot = the body origin */
    float rotation[4];              /* the neutral pose, world: the joint frame, x y z w */
    float twist_axis[3];            /* constraint to the parent, at this pivot, in the neutral pose: twist axis (world; along this part's bone) */
    float plane_axis[3];            /* and a perpendicular axis; made orthogonal to the twist axis if it is not */
    float parent_twist_axis[3];     /* the same two axes as the PARENT holds them (world, neutral pose); all zero -> the part's own.
                                       Limits are centred where the parent's frame meets the part's: rotate these away from the
                                       part's axes and the cones sit off the neutral pose (a knee's cone centred mid-flexion) */
    float parent_plane_axis[3];
    float twist_min_deg;            /* twist range about twist_axis, -180..180 */
    float twist_max_deg;
    float normal_half_cone_deg;     /* swing limits, 0..180: how far the bone may tilt TOWARD the normal (twist x plane) axis,
                                       i.e. a rotation about the plane axis — a hinge's bend when the plane axis is its pin */
    float plane_half_cone_deg;      /* and how far toward the plane axis (a rotation about the normal axis) */
    float motor_frequency;          /* Hz of the motor spring; <= 0 -> 20 */
    float motor_damping;            /* damping ratio; <= 0 -> 2 */
    float max_torque;               /* N m the motor may apply; <= 0 -> 500 */
    float friction_torque;          /* N m of resistance while the motors are off */
    float friction;                 /* the part's surface friction against what it lands on; <= 0 -> 0.5 */
} ujolt_ragdoll_part;

typedef struct ujolt_ragdoll_desc {
    const ujolt_ragdoll_part *parts;
    uint32_t part_count;
    uint64_t user_data;             /* the entity every part reports (contacts, rays) */
    uint32_t layer;                 /* engine collision layer 0..31 for every part */
    float gravity_factor;           /* 1 = normal */
    float linear_damping;           /* <= 0 -> Jolt's default (0.05) */
    float angular_damping;          /* <= 0 -> Jolt's default (0.05) */
    float max_linear_velocity;      /* m/s; <= 0 -> 50 */
    int32_t start_active;           /* 0/1: whether the parts are in the world at creation */
    uint32_t velocity_steps;        /* solver iterations for the joints, per step; 0 -> the world's (Jolt: 10 / 2). A long
                                       powered chain converges slowly: its outer links trail the animation until the joints
                                       get more iterations than the world's default */
    uint32_t position_steps;
    const int32_t *disabled_pairs;  /* pairs of part indices (a0 b0 a1 b1 ...) that never collide with each other, on top of
                                       every parent/child pair and every pair overlapping in the neutral pose; NULL for none.
                                       A torso resting on its own thighs props a fallen body up: disable those pairs */
    uint32_t disabled_pair_count;
} ujolt_ragdoll_desc;

/// NULL for a bad description: no parts, a parent index not before its
/// child, a root that is not part 0 or more than one root, a hull shape,
/// or Jolt out of bodies. The parts start kinematic, motors off.
ujolt_ragdoll *ujolt_world_add_ragdoll(ujolt_world *world, const ujolt_ragdoll_desc *desc);
void ujolt_world_remove_ragdoll(ujolt_world *world, ujolt_ragdoll *ragdoll);
uint32_t ujolt_ragdoll_part_count(const ujolt_ragdoll *ragdoll);
/// Adds the parts and constraints to the world / removes them (they keep
/// their state); idempotent.
void ujolt_ragdoll_set_active(ujolt_ragdoll *ragdoll, int32_t active);
int32_t ujolt_ragdoll_is_active(const ujolt_ragdoll *ragdoll);
/* Poses are one 4x4 rigid world transform per part, 16 floats column-major
   (simd_float4x4 layout), body origin = joint pivot. */
/// Teleports every part; resets the constraints' warm start; wakes the
/// ragdoll when active.
void ujolt_ragdoll_set_pose(ujolt_ragdoll *ragdoll, const float *world_matrices, int32_t reset_velocities);
/// 3 floats per part each; angular may be NULL.
void ujolt_ragdoll_set_velocities(ujolt_ragdoll *ragdoll, const float *linear, const float *angular);
/// The pose the kinematic parts move to during the next step(s), carrying
/// the implied velocity; kept until replaced. Dynamic parts ignore it.
void ujolt_ragdoll_set_kinematic_pose(ujolt_ragdoll *ragdoll, const float *world_matrices);
/// Motor targets (local rotations derived from the pose) for every
/// constraint whose motors are in position mode; consumed by the next step.
void ujolt_ragdoll_drive_motors(ujolt_ragdoll *ragdoll, const float *world_matrices);
/// part -1 = all; dynamic 0 = kinematic (a part made kinematic stops where
/// it is until it is given a pose).
void ujolt_ragdoll_set_part_dynamic(ujolt_ragdoll *ragdoll, int32_t part, int32_t dynamic);
int32_t ujolt_ragdoll_part_is_dynamic(const ujolt_ragdoll *ragdoll, int32_t part);
/** Whether Jolt may put the parts to sleep when they come to rest (its default). A body settling through a slow
    topple can pause below the sleep threshold long enough to be frozen mid-fall; forbid sleeping until it lies. */
void ujolt_ragdoll_set_allow_sleeping(ujolt_ragdoll *ragdoll, int32_t allow);
/** 1 while Jolt simulates the part (awake), 0 once it sleeps or the ragdoll is out of the world. */
int32_t ujolt_ragdoll_part_is_awake(const ujolt_ragdoll *ragdoll, int32_t part);
/// part -1 = all (the root has no constraint and is skipped); values <= 0
/// keep the part's current setting.
void ujolt_ragdoll_set_motors(ujolt_ragdoll *ragdoll, int32_t part, ujolt_motor_mode mode, float frequency, float damping, float max_torque, float friction_torque);
/// Current world transform of every part; returns the count written
/// (capped at capacity).
uint32_t ujolt_ragdoll_read_pose(const ujolt_ragdoll *ragdoll, float *world_matrices, uint32_t capacity);
/// N s at a world point; dynamic parts of an active ragdoll only.
/** N s applied to a dynamic part at a world point, or at its centre of mass when world_point is NULL (a push without spin). */
void ujolt_ragdoll_add_impulse(ujolt_ragdoll *ragdoll, int32_t part, const float impulse[3], const float world_point[3]);
/** Adds m/s to a part's linear velocity (-1 = every part), on top of whatever it has: a shove of the whole body. */
void ujolt_ragdoll_add_linear_velocity(ujolt_ragdoll *ragdoll, int32_t part, const float delta[3]);
/** Gravity on a part (-1 = every part): 1 = the world's, 0 = none. A reacting limb held by the animation feels only the hit. */
void ujolt_ragdoll_set_gravity_factor(ujolt_ragdoll *ragdoll, int32_t part, float factor);
/** The rotation of a part's joint in its constraint space (x y z w): identity where the parent's and the part's
    axes meet, twist about X, swing about Y (the normal axis) and Z (the plane axis). For tuning limits; 0 for the root. */
int32_t ujolt_ragdoll_read_joint_rotation(const ujolt_ragdoll *ragdoll, int32_t part, float out_quat[4]);

#ifdef __cplusplus
}
#endif

#endif /* CJOLTBRIDGE_H */
