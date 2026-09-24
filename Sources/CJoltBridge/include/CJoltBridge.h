//
//  CJoltBridge.h
//  UntoldJoltPhysics
//
//  C API over Jolt Physics for the Untold Engine physics backend plugin.
//  Deliberately narrow: it mirrors the engine's PhysicsBackend protocol
//  (bodies in, kinematic targets in, step, transforms out, buffered events
//  out, one raycast) so the Swift side stays a thin adapter, plus the plugin's
//  own extras (soft bodies, a character controller). All functions except
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
    const uint32_t *faces;         /* index triples; optional */
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

#ifdef __cplusplus
}
#endif

#endif /* CJOLTBRIDGE_H */
