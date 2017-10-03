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
            bool select = true;
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
