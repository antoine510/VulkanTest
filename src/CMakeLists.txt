
file(GLOB SRC_FILES
	"*.cpp"
	"*.h"
)

add_executable(VulkanTest ${SRC_FILES})

target_link_libraries(VulkanTest "SDL2" "SDL2main" "SDL2_image" "vulkan-1")

set_target_properties(VulkanTest
    PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY_RELEASE 			"${CMAKE_SOURCE_DIR}/Binary"
	RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO 	"${CMAKE_SOURCE_DIR}/Binary"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG 				"${CMAKE_SOURCE_DIR}/Binary"
	VS_DEBUGGER_WORKING_DIRECTORY				"${CMAKE_SOURCE_DIR}/Binary"
)
