__constant sampler_t sampler_const =
CLK_NORMALIZED_COORDS_FALSE |
CLK_ADDRESS_NONE |
CLK_FILTER_NEAREST;

__kernel void clove(write_only image2d_t dest) {
	const int2 pixel_id = (int2)(get_global_id(0), get_global_id(1));
	const float4 oout = (float4)(1.0, 1.0, 0.5, 1.0);
	write_imagef(dest, pixel_id, oout);
}
