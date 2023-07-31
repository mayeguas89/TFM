# Looks for the environment variable:
# OPTIX77_PATH

# Sets the variables :
# MAXWELL_INCLUDE_DIR

# MAXWELL_FOUND

set(MAXELL_SDK_PATH "C:/Program Files/Next Limit/Maxwell Render 5/sdk")
set(MXVERSION "x64-v140")
find_path(MAXWELL_INCLUDE_DIR maxwell.h ${MAXELL_SDK_PATH}/include)
find_path(MAXWELL_LIB_DIR mxpublic-${MXVERSION}.lib ${MAXELL_SDK_PATH}/lib/${MXVERSION})

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(MAXWELL DEFAULT_MSG MAXWELL_INCLUDE_DIR MAXWELL_LIB_DIR)

mark_as_advanced(MAXWELL_INCLUDE_DIR MAXWELL_WCOMMON_LIBRARY)

message("MAXWELL_FOUND = " "${MAXWELL_FOUND}")
message("MAXWELL_INCLUDE_DIR = " "${MAXWELL_INCLUDE_DIR}")
message("MAXWELL_LIB_DIR = " "${MAXWELL_LIB_DIR}")
