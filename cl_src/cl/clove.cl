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
float rand_pm(uint *seed) {
    /* return (*seed) / 2147483647.0f; // 2**31-1 */

    // https://stackoverflow.com/questions/9912143/how-to-get-a-random-number-in-opencl
    long const a = 16807; // 7**5
    long const m = 2147483647; // 2**31-1
    *seed = ((long)(*seed) * a) % m;
    /* return (*seed) / 4294967295.0; */
    return (*seed) / 2147483647.0f; // 2**31-1
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

// Pick a new color for pixels on the frontier.
// Mask is hot for pixels that are already filled.
__kernel void inflate(
    read_only image2d_t in_canvas,
    read_only image2d_t in_mask,
    global uint *rand,
    write_only image2d_t out_canvas,
    write_only image2d_t out_mask)
{
    const int2 pixel_id = (int2)(get_global_id(0), get_global_id(1));
    /* const float2 pixel_idf  = (float2)((float)pixel_id.x, (float)pixel_id.y); */
    const uint rand_id = get_global_id(0) + get_global_id(1) * get_global_size(0);
    uint rand_seed = rand[rand_id];
    const int2 dims = get_image_dim(out_canvas);

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

    if (rand_pm(&rand_seed) < 0.05) {
        /* float2 virtual_xy = (float2)(pixel_idf.x / 50, pixel_idf.y / 50); */
        /* const int2 center_xy = (int2)(dims.x / 2, dims.y / 2); */
        /* float factor = distance(convert_float2(pixel_id), convert_float2(center_xy)) / distance((float2)(0, 0), convert_float2(dims)); */
        const int2 offset = (int2)((rand_pm(&rand_seed) - 0.5) * 5,
                                   (rand_pm(&rand_seed) - 0.5) * 5);
        const float4 src_rgba = read_imagef(in_canvas, sampler_const, pixel_id + offset);
        const float4 mask_self = read_imagef(in_mask, sampler_const, pixel_id + offset);
        write_imagef(out_canvas, pixel_id, src_rgba);
        write_imagef(out_mask, pixel_id, mask_self);
        return;
    }

    const float4 src_rgba = read_imagef(in_canvas, sampler_const, pixel_id);
    const float4 mask_self = read_imagef(in_mask, sampler_const, pixel_id);
    // TODO second term (rand off for circle) is wasteful
    if (mask_self.x > .5 || rand_pm(&rand_seed) < 0.8) {
        // This spot is filled already. Copy over and get outta here.
        write_imagef(out_canvas, pixel_id, src_rgba);
        write_imagef(out_mask, pixel_id, mask_self);
        return;
    }

    int n_hot_neighbors = 0;
    // TODO neighbor should be selected randomly
    float4 selected_neighbor_rgba = (float4)(1,0,1,1);
    for (int dx = -1; dx <= 1;  dx++) {
        for (int dy = -1; dy <= 1;  dy++) {
            const int2 loc = pixel_id + (int2)(dx, dy);
            bool self = (dx == 0 && dy == 0);
            bool in_bounds = (loc.x >= 0 && loc.y >= 0 && loc.x < dims.x && loc.y < dims.y);
            /* bool select = true; */
            if (!self && in_bounds) {
                const float4 mask_neighbor = read_imagef(in_mask, sampler_const, loc);
                if (mask_neighbor.x > .5) {
                    n_hot_neighbors += 1;
                    const float4 rgba_neighbor = read_imagef(in_canvas, sampler_const, loc);
		    selected_neighbor_rgba = rgba_neighbor;
                }
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

	/* const float distance = .1; */
	const float distance = .04;
	out_canvas_rgba = color_at_distance(selected_neighbor_rgba, distance, &rand_seed); 

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
