md -Force Binary\shaders > $null
& $Env:VULKAN_SDK\Bin\glslc.exe -fshader-stage=vert -o Binary\shaders\cubevert.spv src\shaders\cube.vert
& $Env:VULKAN_SDK\Bin\glslc.exe -fshader-stage=frag -o Binary\shaders\cubefrag.spv src\shaders\cube.frag
& $Env:VULKAN_SDK\Bin\glslc.exe -fshader-stage=vert -o Binary\shaders\terrainvert.spv src\shaders\terrain.vert
& $Env:VULKAN_SDK\Bin\glslc.exe -fshader-stage=frag -o Binary\shaders\terrainfrag.spv src\shaders\terrain.frag