__constant sampler_t sampler_const =
CLK_NORMALIZED_COORDS_FALSE |
CLK_ADDRESS_NONE |
CLK_FILTER_NEAREST;

__kernel void clove(write_only image2d_t dest) {
    const int2 pixel_id = (int2)(get_global_id(0), get_global_id(1));
    const int2 dims = get_image_dim(dest);
    const float3 a_rgb = (float3)((float)pixel_id.x / dims.x, (float)pixel_id.y / dims.y, 0.5);
    const float factor = max((float).3, ((float)pixel_id.x * (float)pixel_id.y) / (dims.x * dims.y));
    const float3 fin_rgb = factor * a_rgb;
    const float4 fin_rgba = (float4)(fin_rgb.x, fin_rgb.y, fin_rgb.z, 1);
    write_imagef(dest, pixel_id, fin_rgba);
}

__kernel void passthru(read_only image2d_t source, write_only image2d_t dest) {
    const int2 pixel_id = (int2)(get_global_id(0), get_global_id(1));
    float4 rgba = read_imagef(source, sampler_const, pixel_id);
    write_imagef(dest, pixel_id, rgba);
}

__kernel void march_penguins(read_only image2d_t source, write_only image2d_t dest) {
    const int2 pixel_id = (int2)(get_global_id(0), get_global_id(1));
    const int2 dims = get_image_dim(dest);
    float4 rgba = read_imagef(source, sampler_const, pixel_id);
    if (pixel_id.x > 0) {
        // scoot
        float4 rgba_neighbor = read_imagef(source, sampler_const, pixel_id + (int2)(-1,0));
        rgba = (rgba + rgba_neighbor) / 2;
    }
    if (abs(pixel_id.x - 40) <= 10 && abs(pixel_id.y - 40) <= 10) {
        // red
        rgba = (float4)(1,0,0,1);
    }
    write_imagef(dest, pixel_id, rgba);
}

__kernel void life(read_only image2d_t source, write_only image2d_t dest) {
    const int2 pixel_id = (int2)(get_global_id(0), get_global_id(1));
    const int2 dims = get_image_dim(dest);
    const float4 src_rgba = read_imagef(source, sampler_const, pixel_id);
    bool live = src_rgba.y > .5;
    int live_neighbors = 0;
    for (int dx = -1; dx <= 1;  dx++) {
        for (int dy = -1; dy <= 1;  dy++) {
            const int2 loc = pixel_id + (int2)(dx, dy);
            bool self = (dx == 0 && dy == 0);
            bool in_bounds = (loc.x >= 0 && loc.y >= 0 && loc.x < dims.x && loc.y < dims.y);
            /* bool select = (abs(loc.x - 10) < 5) && (abs(loc.y - 10) < 5); */
            /* bool select = true; */
            if (!self && in_bounds) {
                const float4 rgba_neighbor = read_imagef(source, sampler_const, loc);
                if (rgba_neighbor.y > .5) {
                    live_neighbors += 1;
                }
            }
        }
    }
    /* if (live_neighbors > 0) { */
    /* 	printf("px: %d %d: live:%d neighbors:%d\n", pixel_id.x, pixel_id.y, */
    /* 	       live, live_neighbors); */
    /* } */
    if (live_neighbors == 3 || (live && live_neighbors == 2)) {
        live = true;
    } else {
	live = false;
    }
    float4 dest_rgba = (float4)(.2,0,0,1);
    if (live) {
        dest_rgba = (float4)(0,1,1,1);
    }
    /* if (abs(pixel_id.x - 5) <= 2 && abs(pixel_id.y - 5) <= 2) { */
    /*     dest_rgba = (float4)(1,1,1,1); */
    /* } */
    write_imagef(dest, pixel_id, dest_rgba);
    /* if (pixel_id.x < 0) { */
    /* 	// Should neve happens. Just here to work around some bug. */
    /*     printf("px: %f %f %f %f\n", src_rgba.x, src_rgba.y, src_rgba.z, src_rgba.w); */
    /* } */
}

/* __kernel void life(read_only image2d_t source, write_only image2d_t dest) { */
/*     const int2 pixel_id = (int2)(get_global_id(0), get_global_id(1)); */
/*     const int2 dims = get_image_dim(dest); */
/*     const float4 src_rgba = read_imagef(source, sampler_const, pixel_id); */
/*     bool live = src_rgba.y > .5; */
/*     for (int dx = 0; dx < 1;  dx++) { */
/*         if (pixel_id.x >= 0) { */
/*             /\* read_imagef(source, sampler_const, (int2)(0,0)); *\/ */
/*             get_image_dim(dest); */
/*             /\* get_global_id(0); *\/ */
/*         } */
/*     } */
/*     float4 dest_rgba = (float4)(.2,0,0,1); */
/*     if (live) { */
/*         dest_rgba = (float4)(0,1,1,1); */
/*     } */
/*     write_imagef(dest, pixel_id, dest_rgba); */
/*     if (pixel_id.x < 0) { */
/*         printf("px: %f %f %f %f\n", src_rgba.x, src_rgba.y, src_rgba.z, src_rgba.w); */
/*     } */
/* } */

// How different are these colors?
// Returns [0, 1] where 0 is most similar.
float color_distance(float4 rgba1, float4 rgba2) {
    /* const float r = rgba1.x - rgba2.x; */
    /* const float g = rgba1.y - rgba2.y; */
    /* const float b = rgba1.z - rgba2.z; */
    /* return (r * r + g * g + b * b) / 3; */

    const float r = fabs(rgba1.x - rgba2.x);
    const float g = fabs(rgba1.y - rgba2.y);
    const float b = fabs(rgba1.z - rgba2.z);
    return (r + g + b ) / 3; 
}

// Score each location based on how close its neighbors are to the goal color.
// Only score against pixels that are white on the mask.
__kernel void score(
    read_only image2d_t source,
    read_only image2d_t mask,
    float4 goal,
    write_only image2d_t dest)
{
    const int2 pixel_id = (int2)(get_global_id(0), get_global_id(1));
    const int2 dims = get_image_dim(dest);
    /* const float4 src_rgba = read_imagef(source, sampler_const, pixel_id); */

    // Whether to use average of minimum of the neighbors as the score.
    const bool alg_avg = false;

    float acc = 0;
    float min_neighbor_score = INFINITY;
    int n_scored_neighbors = 0;
    for (int dx = -1; dx <= 1;  dx++) {
        for (int dy = -1; dy <= 1;  dy++) {
            const int2 loc = pixel_id + (int2)(dx, dy);
            bool self = (dx == 0 && dy == 0);
            bool in_bounds = (loc.x >= 0 && loc.y >= 0 && loc.x < dims.x && loc.y < dims.y);
            /* bool select = true; */
            if (!self && in_bounds) {
                const float4 mask_neighbor = read_imagef(mask, sampler_const, loc);
                if (mask_neighbor.x > .5) {
                    const float4 rgba_neighbor = read_imagef(source, sampler_const, loc);
                    n_scored_neighbors += 1;
                    const float neighbor_score = color_distance(goal, rgba_neighbor);
                    acc += neighbor_score;
                    if (neighbor_score < min_neighbor_score) {
                        min_neighbor_score = neighbor_score;
                    }
                }
            }
        }
    }
    if (n_scored_neighbors == 0) {
        acc = INFINITY;
    } else {
        acc /= n_scored_neighbors;
    }
    float dest_val = min_neighbor_score;
    if (alg_avg) {
        dest_val = acc;
    }
    float4 dest_rgba = (float4)(dest_val, 0, 0, 1);
    write_imagef(dest, pixel_id, dest_rgba);
    /* if (live_neighbors > 0) { */
    /* 	printf("px: %d %d: live:%d neighbors:%d\n", pixel_id.x, pixel_id.y, */
    /* 	       live, live_neighbors); */
    /* } */
}

// this one is really bad
float rand_xorshift(uint2 randoms, uint globalID) {
    // https://stackoverflow.com/questions/9912143/how-to-get-a-random-number-in-opencl
    uint seed = randoms.x + globalID;
    uint t = seed ^ (seed << 11);  
    uint result = randoms.y ^ (randoms.y >> 19) ^ (t ^ (t >> 8));
    return (float)result / 4294967295.0;
}

// 1 <= *seed < m
uint rand_pm_uint(uint *seed) {
    /* return (*seed) / 2147483647.0f; // 2**31-1 */

    // https://stackoverflow.com/questions/9912143/how-to-get-a-random-number-in-opencl
    long const a = 16807; // 7**5
    long const m = 2147483647; // 2**31-1
    *seed = ((long)(*seed) * a) % m;
    /* return (*seed) / 4294967295.0; */
    return *seed;
}

// 1 <= *seed < m
float rand_pm(uint *seed) {
    return rand_pm_uint(seed) / 2147483647.0f; // 2**31-1
}

// Find a new color that is different from `rgba1` by `d`.
// d is [0, 1] where 0 is most similar.
// TODO this is so wrong
float4 color_at_distance(float4 rgba1, float d, uint *rand_seed) {
    float3 delta = (float3)(rand_pm(rand_seed), rand_pm(rand_seed), rand_pm(rand_seed));
    /* delta = (float3)(.5,.5,.5); */
    /* printf("%f %f %f\n", delta.x, delta.y, delta.z); */
    /* float rx = rand_pm(rand_seed); */
    /* float3 delta = (float3)(rx, rx, rx); */
    /* return (float4)(delta.x, delta.x, delta.x, 1); */

    delta -= (float3)(.5, .5, .5);
    /* delta -= (float3)(.1, .1, .1); */
    /* delta -= (float3)(.1, .1, .1); */
    /* delta -= (float3)(.1, .1, .1); */
    /* delta -= (float3)(.1, .1, .1); */
    /* delta = (float3)(-10, -10, 2); */
    delta = normalize(delta) * d;
    const float r = rgba1.x + delta.x;   
    const float g = rgba1.y + delta.y; 
    const float b = rgba1.z + delta.z; 
    /* return (float4)(r, g, b, 1); */
    float4 result = (float4)(r, g, b, rgba1.w);
    /* printf("%f\n", rgba1.z); */
    /* result = (float4)(0,0,0,1); */
    return clamp(result, 0.0f, 1.0f);
}

float4 color_avg(float4 rgba1, float4 rgba2) {
    return (rgba1 + rgba2) / 2;
}

float4 color_mix(float factor, float4 c1, float4 c2) {
    factor = clamp(factor, 0.0f, 1.0f);
    return (1.0f - factor) * c1 + factor * c2;
}

__constant int2 neighbor_deltas[8] = {
    (int2)(-1, -1),
    (int2)(-1, 0),
    (int2)(-1, 1),
    (int2)(0, -1),
    // skip 0, 0
    (int2)(0, 1),
    (int2)(1, -1),
    (int2)(1, 0),
    (int2)(1, 1),
};

// Pick a new color for pixels on the frontier.
// Mask is hot for pixels that are already filled.
__kernel void pastiche(
    read_only image2d_t in_canvas,
    read_only image2d_t in_mask,
    read_only image2d_t in_subject,
    global uint *rand,
    read_only uint time_ms,
    read_only uint cursor_enabled,
    read_only uint2 cursor_xy,
    write_only image2d_t out_canvas,
    write_only image2d_t out_mask)
{
    const int2 pixel_id = (int2)(get_global_id(0), get_global_id(1));
    /* const float2 pixel_idf  = (float2)((float)pixel_id.x, (float)pixel_id.y); */
    const uint rand_id = get_global_id(0) + get_global_id(1) * get_global_size(0);
    uint rand_seed = rand[rand_id];
    const int2 dims = get_image_dim(out_canvas);

    // Show the subject
    /* const float4 subject_rgba = read_imagef(in_subject, sampler_const, pixel_id);  */
    /* write_imagef(out_canvas, pixel_id, subject_rgba);    */
    /* return;  */

    /* // test of randomness */
    /* {  */
    /*     float rx1 = rand_pm(&rand_seed);  */
    /*     float rx2 = rand_pm(&rand_seed);  */
    /*     /\* if (pixel_id == (int2)(100, 100)) { *\/  */
    /*     if (pixel_id.x == 100 && pixel_id.y == 100) {  */
    /*         printf("%f, %f\n", rx1, rx2); */
    /*     }  */
    /*     float rx = ((rx1 - rx2) + 1) / 2;  */
    /*     float4 out_canvas_rgba = (float4)(rx, rx, rx, 1);  */
    /*     write_imagef(out_canvas, pixel_id, out_canvas_rgba);  */
    /*     return;  */
    /* }  */

    // Clear some pixels sometimes
    /* if (rand_pm(&rand_seed) < 0.1) { */
    /*     const float4 src_rgba = read_imagef(in_canvas, sampler_const, pixel_id); */
    /*     float4 out_mask_rgba = (float4)(0, 0, 0, 1); */
    /*     write_imagef(out_canvas, pixel_id, src_rgba); */
    /*     write_imagef(out_mask, pixel_id, out_mask_rgba); */
    /*     return; */
    /* } */

    if (cursor_enabled > 0 && time_ms > 4000) {
        const float distance_to_cursor = distance(convert_float2(pixel_id), convert_float2(cursor_xy));
        if (distance_to_cursor < 40 + (20 * rand_pm(&rand_seed))) {
            /* const int2 offset = (int2)((rand_pm(&rand_seed) - 0.5) * 3 + 4, */
            /*                         (rand_pm(&rand_seed) - 0.5) * 2 + 2); */
	    const int2 offset = (int2)(0,0);
            const float4 src_rgba = read_imagef(in_canvas, sampler_const, pixel_id + offset);
            /* const float4 out_canvas_rgba = (float4)(0, 0, 0, 1); */
            const float4 out_mask_rgba = (float4)(0, 0, 0, 1);
            write_imagef(out_canvas, pixel_id, src_rgba);
            write_imagef(out_mask, pixel_id, out_mask_rgba);
            return;
        }
    }

    // Do the fizzy thing where pixels wiggle around.
    if (rand_pm(&rand_seed) < 0.05) {
        /* float2 virtual_xy = (float2)(pixel_idf.x / 50, pixel_idf.y / 50); */
        /* const int2 center_xy = (int2)(dims.x / 2, dims.y / 2); */
        /* float factor = distance(convert_float2(pixel_id), convert_float2(center_xy)) / distance((float2)(0, 0), convert_float2(dims)); */
        const int2 offset = (int2)((rand_pm(&rand_seed) - 0.5) * 3,
                                   (rand_pm(&rand_seed) - 0.5) * 2);
        const float4 src_rgba = read_imagef(in_canvas, sampler_const, pixel_id + offset);
        const float4 mask_self = read_imagef(in_mask, sampler_const, pixel_id + offset);
        write_imagef(out_canvas, pixel_id, src_rgba);
        write_imagef(out_mask, pixel_id, mask_self);
        return;
    }

    const float4 src_rgba = read_imagef(in_canvas, sampler_const, pixel_id);
    const float4 mask_self = read_imagef(in_mask, sampler_const, pixel_id);
    // TODO second term (rand off for circle) is wasteful
    if (mask_self.x > .5 || rand_pm(&rand_seed) < 0.7) {
        // This spot is filled already. Copy over and get outta here.
        write_imagef(out_canvas, pixel_id, src_rgba);
        write_imagef(out_mask, pixel_id, mask_self);
        return;
    }

    int n_hot_neighbors = 0;
    // TODO neighbor should be selected randomly
    float4 selected_neighbor_rgba = (float4)(1,0,1,1);
    int neighbor_index_offset = rand_pm_uint(&rand_seed) % 8;
    for (int i = 0; i < 8; i++) {
        int2 dxy = neighbor_deltas[(i + neighbor_index_offset) % 8];

        const int2 loc = pixel_id + dxy;
        bool in_bounds = (loc.x >= 0 && loc.y >= 0 && loc.x < dims.x && loc.y < dims.y);
        /* bool select = true; */
        if (in_bounds) {
            const float4 mask_neighbor = read_imagef(in_mask, sampler_const, loc);
            if (mask_neighbor.x > .5) {
                n_hot_neighbors += 1;
                const float4 rgba_neighbor = read_imagef(in_canvas, sampler_const, loc);
                selected_neighbor_rgba = rgba_neighbor;
            }
        }
    }

    // Whether we are on the frontier
    const bool self_frontier = n_hot_neighbors > 0;

    float4 out_canvas_rgba = src_rgba;
    float4 out_mask_rgba = (float4)(0, 0, 0, 1);
    if (self_frontier) {
        /* out_canvas_rgba = (float4)(0, 1, .2, 1); */
	/* out_canvas_rgba = selected_neighbor_rgba; */
	// TODO distance should be selected randomly (along a curve representative of original)
	/* const float distance = rand_pm(&rand_seed); */
        /* out_canvas_rgba = (float4)(0, .5, rand_pm(&rand_seed), 1); */

	const float4 subject_rgba = read_imagef(in_subject, sampler_const, pixel_id);

	const float distance = .04;
	/* const float distance = .01; */
	/* const float distance = .005; */
	/* const float distance = cos(convert_float(time_ms) * 0.0001) * .08f; */
	/* const float distance = cos(convert_float(pixel_id.x) * 0.004) * .08f; */
	/* out_canvas_rgba = color_at_distance(selected_neighbor_rgba, distance, &rand_seed); */
        /* const float factor = 0.02 + 0.02 * -cos(convert_float(time_ms / 3000)); */
        /* const float factor = 0.1 * (1.0f - length(subject_rgba) / 3); */
        float max4len = length((float4)(1, 1, 1, 1));
        /* if (length(subject_rgba) / max4len > 0.7f) { */
        /*     factor = 0.0f; */
        /* } */
        /* float factor = 0.08f * (1.0f - length(subject_rgba) / max4len); */
        /* float factor = 0.04f; */
        float factor = 0.1f;
        const float4 departure_lounge = color_mix(factor, selected_neighbor_rgba, subject_rgba);
	out_canvas_rgba = color_at_distance(departure_lounge, distance, &rand_seed);

	/* /\* Write out the subject. *\/ */
	/* if (rand_pm(&rand_seed) < 0.05) { */
	/*     const float4 subject_rgba = read_imagef(in_subject, sampler_const, pixel_id);     */
	/*     out_canvas_rgba = subject_rgba;  */
	/* }  */

	/* Though must be close to elephant */
	/* for (int i = 0; i < 100; i++) { */
	/*     if (color_distance(subject_rgba, out_canvas_rgba) > 0.1) { */
	/* 	// Re-reoll */
	/* 	out_canvas_rgba = color_at_distance(selected_neighbor_rgba, distance, &rand_seed); */
	/*     } else { */
	/*       break; */
	/*     } */
	/* } */

	/* const float rx = rand_pm(&rand_seed); */
	/* out_canvas_rgba = (float4)(rx, rx, rx, 1); */

        /* const float xjdf = rand_pm(&rand_seed); */
        /* out_canvas_rgba = (float4)(xjdf, xjdf, xjdf, 1.0); */
        /* out_canvas_rgba = (float4)(rand_pm(&rand_seed), rand_pm(&rand_seed), rand_pm(&rand_seed), 1); */
        out_mask_rgba = (float4)(1, 1, 1, 1);
    }
    write_imagef(out_canvas, pixel_id, out_canvas_rgba);
    write_imagef(out_mask, pixel_id, out_mask_rgba);
}
