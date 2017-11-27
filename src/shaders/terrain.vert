#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (binding = 0) buffer TerrainBuffer {
	//blocs: 0x00|intensity|backwallTile|blocTile
	//blocTile/backwallTile: 0xV|U
	uint blocs[4096];
} terrainBuffer;

layout(push_constant) uniform VertexPush {
	//transform: (camXbloc, camYbloc, sizeX, sizeY)
    layout(offset = 0) vec4 transform;
	layout(offset = 28) bool backwall;
} vertexPush;

//vdata: 0x000|spot|coordY|coordX
layout (location = 0) in uint vdata;
layout (location = 0) out vec2 outUV;
layout (location = 1) out float intensity;

void main() {
	uint xCoord = vdata & 0xff;
	uint yCoord = (vdata >> 8) & 0xff;
	const bool right = (vdata & 0x10000) != 0;
	const bool down = (vdata & 0x20000) != 0;
	const uint bIndex = xCoord + (yCoord << 6);
	const uint blocType = terrainBuffer.blocs[bIndex];
	const uint blocTile = (blocType >> (vertexPush.backwall ? 8 : 0)) & 0xff;
	uint texU = blocTile & 0xf, texV = blocTile >> 4;
	if(right) {
		xCoord++; texU++;
	}
	if(down) {
		yCoord++; texV++;
	}
	
	outUV = vec2(texU, texV) / 16.f;
	intensity = ((blocType >> 16) & 0xff) / 255.0f;
	gl_Position = vec4((vec2(xCoord, yCoord) + vertexPush.transform.xy) * vertexPush.transform.zw, 0, 1);
}