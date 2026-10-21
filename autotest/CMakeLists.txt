# THIS SOFTWARE WAS DEVELOPED AT THE ROCKY MOUNTAIN RESEARCH STATION (RMRS)
# MISSOULA FIRE SCIENCES LABORATORY BY EMPLOYEES OF THE FEDERAL GOVERNMENT
# IN THE COURSE OF THEIR OFFICIAL DUTIES. PURSUANT TO TITLE 17 SECTION 105
# OF THE UNITED STATES CODE, THIS SOFTWARE IS NOT SUBJECT TO COPYRIGHT
# PROTECTION AND IS IN THE PUBLIC DOMAIN. RMRS MISSOULA FIRE SCIENCES
# LABORATORY ASSUMES NO RESPONSIBILITY WHATSOEVER FOR ITS USE BY OTHER
# PARTIES,  AND MAKES NO GUARANTEES, EXPRESSED OR IMPLIED, ABOUT ITS QUALITY,
# RELIABILITY, OR ANY OTHER CHARACTERISTIC.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

#cmake_minimum_required(VERSION 3.16)

include_directories(${PROJECT_SOURCE_DIR}/src/ninja
                    ${NETCDF_INCLUDES}
                    ${GDAL_SYSTEM_INCLUDE} ${GDAL_INCLUDE_DIR}
                    ${CURL_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS})

set(LINK_LIBS ${NETCDF_LIBRARIES_C}
              ${GDAL_LIBRARY}
              ${CURL_LIBRARIES}
              ${Boost_LIBRARIES})

# link the target, not the library file, so its dependencies (shapelib,
# OpenMP, hwloc) come with it
set(LINK_LIBS ${LINK_LIBS} ninja)

set(TEST_SOURCES test_main.cpp
                 test_contain.cpp
                 test_srtm.cpp
                 test_gdal_fetch.cpp
                 test_gdal_output.cpp
                 test_gdal_util.cpp
                 test_grid_interp.cpp
                 test_array2d.cpp
                 test_timezone.cpp
                 test_init.cpp
                 #test_input_points.cpp
                 test_buffer_grid.cpp
                 test_ninjaunits.cpp
                 test_stl.cpp
                 test_rmtree.cpp
                 momentum_compare.cpp
                 test_momentum_compare.cpp
                 test_rans_grid.cpp
                 test_rans_solver.cpp
                 test_utm.cpp)
if(WITH_LCP_CLIENT)
    set(TEST_SOURCES ${TEST_SOURCES} test_landfireclient.cpp)
endif(WITH_LCP_CLIENT)

if(ENABLE_GMTED)
    set(TEST_SOURCES ${TEST_SOURCES} test_gmted.cpp)
endif(ENABLE_GMTED)

if(WITH_NOMADS_SUPPORT)
    set(TEST_SOURCES ${TEST_SOURCES}
                     test_simplenomadsclient.cpp
                     test_utctime.cpp)
endif(WITH_NOMADS_SUPPORT)

if(NINJAFOAM)
    set(TEST_SOURCES ${TEST_SOURCES}
                     test_foam.cpp
                     test_momentum_backend.cpp)
endif(NINJAFOAM)

add_executable(test_main ${TEST_SOURCES})

add_dependencies(test_main ninja)
#ADD_DEPENDENCIES(test_api  ninja)

target_link_libraries(test_main ${LINK_LIBS})
target_compile_definitions(test_main PRIVATE
                           WINDNINJA_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

# Compare momentum solver output against autotest/golden/momentum
add_executable(compare_momentum compare_momentum.cpp momentum_compare.cpp)
target_link_libraries(compare_momentum ${GDAL_LIBRARY})

set(TEST_API FALSE)
if(TEST_API)
    add_executable( test_api test_capi.c )
    target_link_libraries(test_api  ninja)
endif(TEST_API)

enable_testing()

if(TEST_API)
    add_test(test_capi test_api)
endif(TEST_API)

# contain Test Suite
add_test(test_contain_point
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=contain/points )
add_test(test_contain_polygon
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=contain/polygon )
add_test(test_contain_polygon2
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=contain/polygon2 )

# grid_interp Test Suite
add_test(test_grid_interp_order
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=grid_interp/order )

# array2d Test Suite
add_test(test_array2d_constructor
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=array2d/constructor )

# timezone Test Suite
add_test(test_timezone_boise
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=timezones/boise )

# init Test Suite
add_test(test_init_gdal
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=init/gdal )

# buffer_grid Test Suite
add_test(test_buffer_grid_init
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=buffer_grid/init_and_set)

# ninjaUnits Test Suite
add_test(test_ninja_units
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=base_units/length_to_base_units)

# gdal_util
add_test(test_gdal_util_center
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=gdal_util/center)

# landfireclient Test Suite - still experimental
if(WITH_LCP_CLIENT)
    add_test(test_landfireclient_extract
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=landfireclient/unzip)
endif(WITH_LCP_CLIENT)

# input_points Test Suite
if(NOT NINJA_GUI)
    # This test doesn't work
    #add_test(test_input_points_mackay
    #         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=input_points/mackay )
endif(NOT NINJA_GUI)

if(NINJA_GDAL_OUTPUT)
add_test(test_gdal_output_shapefile
    ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=gdal_output/shapefile)

add_test(test_gdal_output_write_arrow
    ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=gdal_output/write_arrow)

add_test(test_gdal_output_arrow_west
    ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=gdal_output/arrow_west)

add_test(test_gdal_output_kml
    ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=gdal_output/kml)

add_test(test_gdal_output_kmz
    ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=gdal_output/kmz)

add_test(test_gdal_output_kmz_color
    ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=gdal_output/kmz_color)
endif(NINJA_GDAL_OUTPUT)

add_test(test_utm_boise
    ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=utm/boise)

add_test(test_utm_adelaide
    ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=utm/adelaide)

# ******************************************************************************
# Slow test section
# ******************************************************************************
if(RUN_SLOW_TESTS)
    # strm Test Suite
    # FIXME: Disable due to compilation issues.
    if(NOT WIN32)
        add_test(test_srtm_us_box
                 ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=srtm/us_box )
        add_test(test_srtm_world_pt
                 ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=srtm/world_point )
        #add_test(test_srtm_gdal
        #         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=srtm/gdal )

        #gmted Test Suite
        if(ENABLE_GMTED)
            add_test(test_gmted_us
                     ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=gmted/us )
            add_test(test_gmted_world
                     ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=gmted/world )
        endif(ENABLE_GMTED)
        add_test(test_gdal_fetch_us_box
                 ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=gdal_fetch/us_box )
    endif(NOT WIN32)

    if(WITH_LCP_CLIENT)
        add_test(test_landfireclient_download_conus
                 ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=landfireclient/mackay )
        add_test(test_landfireclient_download_alaska
                 ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=landfireclient/alaska )
        add_test(test_landfireclient_download_alaska_bad_geom
                 ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=landfireclient/alaska_bad_geom)
        add_test(test_landfireclient_download_alaska_out
                 ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=landfireclient/alaska_out)
        add_test(test_landfireclient_download_hawaii
                 ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=landfireclient/hawaii )
    endif(WITH_LCP_CLIENT)
endif(RUN_SLOW_TESTS)
if(WITH_NOMADS_SUPPORT)
    add_test(nomads_nam_conus_3_hour_zip
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 mackay nam_conus 3 4 0 zip)
    add_test(nomads_nam_conus_0_hour_zip
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 mackay nam_conus 0 1 0 zip)
    add_test(nomads_rap_mackay_18_limit_zip
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 mackay rap_conus 18 19 0 zip)
    add_test(nomads_rap_north_america_1_zip
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 mackay rap_north_america 1 2 0 zip)
    add_test(nomads_nam_north_america_1_hour_zip
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 mackay nam_north_america 1 1 0 zip)
    add_test(nomads_rap_1_hour_zip
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 mackay rap_conus 1 2 0 zip)
    add_test(nomads_gfs_mackay_zip
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 mackay gfs_global 1 1 0 zip)
    add_test(nomads_gfs_alaska_zip
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 alaska gfs_global 1 1 0 zip)
    add_test(nomads_gfs_africa_zip
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 africa gfs_global 1 1 0 zip)
    add_test(nomads_gfs_south_america_zip
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 south_america gfs_global 1 1 0 zip)
    add_test(nomads_gfs_mackay_large_zip
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 mackay_large gfs_global 12 5 0 zip)
    add_test(nomads_nam_alaska_1_hour_zip
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 alaska nam_alaska 1 2 0 zip)
    add_test(nomads_hires_arw_conus_1_hour_zip
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 mackay hires_arw_conus 1 2 0 zip)
    add_test(nomads_hires_fv3_conus_1_hour_zip
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 mackay hires_fv3_conus 1 2 0 zip)
    add_test(nomads_hires_arw_ak_1_hour_zip
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 mackay hires_arw_alaska 1 2 0 zip)
    add_test(nomads_hires_fv3_ak_1_hour_zip
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 mackay hires_fv3_alaska 1 2 0 zip)
    add_test(nomads_hrrr_1_hour_zip
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 mackay hrrr_conus 1 2 0 zip)
    add_test(nomads_hrrr_ak_1_hour_zip
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 alaska hrrr_alaska 1 2 0 zip)
    add_test(nomads_hrrr_15_hour_zip
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 mackay hrrr_conus 15 16 0 zip)
    add_test(nomads_hrrr_ext_1_hour_zip
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 mackay hrrr_conus_ext 1 2 0 zip)
    add_test(nomads_buffer_1
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 small gfs_global 1 2 0 zip)
    if(NOMADS_EXPER_FORECASTS)
        add_test(nomads_nam_nest_conus_1_hour_zip
                 ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 mackay nam_nest_conus 1 2 0 zip)
        add_test(nomads_nest_conus_1_hour_zip
                 ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 mackay nam_nest_conus 1 2 0 zip)
        add_test(nomads_nest_alaska_1_hour_zip
                 ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 mackay nam_nest_alaska 1 2 0 zip)
        add_test(nomads_nam_narre_1_hour_zip
                 ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 mackay narre 1 2 0 zip)
    endif(NOMADS_EXPER_FORECASTS)
    if(NOMADS_RTMA)
        add_test(nomads_rtma_conus_1_hour_zip
                 ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/download_1 mackay rtma_conus 1 1 0 zip)
    endif(NOMADS_RTMA)
    # UTC
    add_test(utc_create_1
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=utc/create_1)
    add_test(utc_add_hours_1
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=utc/add_hours_1)
    add_test(utc_add_hours_2
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=utc/add_hours_2)
    add_test(utc_add_hours_3
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=utc/add_hours_3)
    add_test(utc_add_hours_4
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=utc/add_hours_4)
    add_test(utc_add_hours_5
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=utc/add_hours_5)
    add_test(utc_add_hours_6
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=utc/add_hours_6)
    add_test(utc_add_hours_7
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=utc/add_hours_7)
    add_test(utc_add_hours_8
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=utc/add_hours_8)
    add_test(utc_add_hours_9
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=utc/add_hours_9)
    add_test(utc_now_1
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=utc/now_1)
    add_test(utc_compare_1
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=utc/compare_1)
    add_test(utc_compare_2
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=utc/compare_2)
    add_test(utc_compare_3
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=utc/compare_3)
    add_test(utc_timet_1
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=utc/from_timet_1)
    add_test(utc_iso_1
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=utc/from_iso_1)
    add_test(utc_iso_invalid_1
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=utc/from_iso_invalid_1)
    add_test(utc_strftime_1
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=utc/strftime_1)
    add_test(utc_copy_1
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=utc/copy_1)
    add_test(utc_copy_2
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=utc/copy_2)
    add_test(form_name_1
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=simplenomadsclient/form_name_1)
endif(WITH_NOMADS_SUPPORT)

if(NINJAFOAM)
    add_test(ninjafoam_compute_direction_axes
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=ninjafoam/compute_direction_axes)
    add_test(ninjafoam_compute_direction_quadrants
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=ninjafoam/compute_direction_quadrants)
    add_test(ninjafoam_set_inlets_axes
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=ninjafoam/set_inlets_axes)
    add_test(ninjafoam_set_inlets_quadrants
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=ninjafoam/set_inlets_quadrants)
    add_test(ninjafoam_replace_key
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=ninjafoam/replace_key)
    add_test(ninjafoam_replace_keys
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=ninjafoam/replace_keys)
    add_test(ninjafoam_is_momentum_solver
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=ninjafoam/is_momentum_solver)
    if(NOT WIN32)
        add_test(ninjafoam_unknown_version
                 ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=ninjafoam/unknown_version)
    endif()
    add_test(momentum_backend_problem_from_inputs
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=momentum_backend/problem_from_inputs)
    add_test(momentum_backend_backend_default_openfoam
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=momentum_backend/backend_default_openfoam)
    add_test(momentum_backend_default_iterations
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=momentum_backend/default_iterations)
    add_test(momentum_backend_backend_names
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=momentum_backend/backend_names)
    add_test(momentum_backend_backend_unavailable_throws
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=momentum_backend/backend_unavailable_throws)
    add_test(momentum_backend_c_api_set_get_backend
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=momentum_backend/c_api_set_get_backend)
    add_test(stl_1
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=stl/stl_1)
    add_test(stl_2
             ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=stl/stl_2)
endif(NINJAFOAM)

add_test(rans_grid_flat_box
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=rans_grid/flat_box)
add_test(rans_grid_closed_surface
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=rans_grid/closed_surface)
add_test(rans_grid_sine_hill
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=rans_grid/sine_hill)
add_test(rans_grid_wall_distance
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=rans_grid/wall_distance)
add_test(rans_grid_from_mesh
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=rans_grid/from_mesh)
add_test(rans_grid_bad_input
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=rans_grid/bad_input)
add_test(rans_solver_freestream
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=rans_solver/freestream)
add_test(rans_solver_mms_laminar
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=rans_solver/mms_laminar)
add_test(momentum_compare_identity
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=momentum_compare/identity)
add_test(momentum_compare_identity_golden_case
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=momentum_compare/identity_golden_case)
add_test(momentum_compare_known_offset
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=momentum_compare/known_offset)
add_test(momentum_compare_known_rotation
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=momentum_compare/known_rotation)
add_test(momentum_compare_circular_difference_wraps
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=momentum_compare/circular_difference_wraps)
add_test(momentum_compare_relative_rmse
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=momentum_compare/relative_rmse)
add_test(momentum_compare_no_data_skipped
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=momentum_compare/no_data_skipped)
add_test(momentum_compare_mismatched_size_throws
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=momentum_compare/mismatched_size_throws)
add_test(momentum_compare_no_valid_cells_throws
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=momentum_compare/no_valid_cells_throws)
add_test(rmtree_1
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=rmtree/rmtree_1)
add_test(rmtree_sym_1
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=rmtree/rmtree_sym_1)
add_test(rmtree_sym_2
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=rmtree/rmtree_sym_2)
add_test(rmtree_sym_3
         ${EXECUTABLE_OUTPUT_PATH}/test_main --run_test=rmtree/rmtree_sym_3)
if(RUN_CFG_TESTS)
    message(STATUS "Generating cli tests...")
    file(GLOB cfgtests ${PROJECT_SOURCE_DIR}/autotest/config/pass/*.cfg)
    message(STATUS "GLOB: ${cfgtests}")
    file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/autotest/config)
    file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/autotest/config/pass)
    foreach(cfgtest ${cfgtests})
        get_filename_component(base ${cfgtest} NAME_WE)
        file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/autotest/config/pass/${base})
        file(COPY ${cfgtest} DESTINATION ${PROJECT_BINARY_DIR}/autotest/config/pass/${base})
        message(STATUS "COMMAND ${PROJECT_BINARY_DIR}/autotest/pc ${PROJECT_BINARY_DIR}/autotest/config/pass/${base}/${base}.cfg elevation_file")
        execute_process(WORKING_DIRECTORY . COMMAND python ${PROJECT_SOURCE_DIR}/autotest/config/pef.py ${PROJECT_BINARY_DIR}/autotest/config/pass/${base}/${base}.cfg elevation_file
                                OUTPUT_VARIABLE ELEV_FILE
                                OUTPUT_STRIP_TRAILING_WHITESPACE)
        string(STRIP ${ELEV_FILE} ELEV_FILE)
        file(COPY ${PROJECT_SOURCE_DIR}/data/${ELEV_FILE} DESTINATION
             ${PROJECT_BINARY_DIR}/autotest/config/pass/${base})
        message(STATUS "Copying elevation file: ${ELEV_FILE}")
        add_test(NAME cfg_${base} COMMAND ${PROJECT_BINARY_DIR}/src/cli/WindNinja_cli ${PROJECT_BINARY_DIR}/autotest/config/pass/${base}/${base}.cfg
                 WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/autotest/config/pass/${base})
        set_tests_properties(cfg_${base} PROPERTIES ENVIRONMENT
                             "WINDNINJA_DATA=${PROJECT_SOURCE_DIR}/data")
        message(STATUS "Adding CLI test cfg_${base} for ${cfgtest}")
    endforeach(cfgtest ${cfgtests})
endif(RUN_CFG_TESTS)

# Momentum solver cfg tests: run the CLI and sanity-check the ascii output.
# Skipped (exit 77) unless an OpenFOAM environment is loaded.
if(RUN_CFG_TESTS AND NINJAFOAM)
    find_program(UV_EXECUTABLE uv)
endif()
if(RUN_CFG_TESTS AND NINJAFOAM AND NOT UV_EXECUTABLE)
    message(WARNING "uv not found; momentum solver cfg tests will not be added")
elseif(RUN_CFG_TESTS AND NINJAFOAM)
    file(GLOB momentumtests ${PROJECT_SOURCE_DIR}/autotest/config/momentum/*.cfg)
    foreach(cfgtest ${momentumtests})
        get_filename_component(base ${cfgtest} NAME_WE)
        set(testdir ${PROJECT_BINARY_DIR}/autotest/config/momentum/${base})
        file(MAKE_DIRECTORY ${testdir})
        file(COPY ${cfgtest} DESTINATION ${testdir})
        file(STRINGS ${cfgtest} elev_line REGEX "^elevation_file")
        string(REGEX REPLACE "^elevation_file *= *([^ ]+).*" "\\1" elev_file "${elev_line}")
        file(COPY ${PROJECT_SOURCE_DIR}/data/${elev_file} DESTINATION ${testdir})
        add_test(NAME cfg_${base}
                 COMMAND ${UV_EXECUTABLE} run --script
                         ${PROJECT_SOURCE_DIR}/autotest/config/momentum/run_momentum_cfg.py
                         ${PROJECT_BINARY_DIR}/src/cli/WindNinja_cli ${testdir}/${base}.cfg
                 WORKING_DIRECTORY ${testdir})
        set_tests_properties(cfg_${base} PROPERTIES
                             ENVIRONMENT "WINDNINJA_DATA=${PROJECT_SOURCE_DIR}/data"
                             SKIP_RETURN_CODE 77)
    endforeach(cfgtest ${momentumtests})
endif()

# Asking for a momentum backend that isn't built in must fail cleanly, before
# any solve (so it doesn't need OpenFOAM).
if(RUN_CFG_TESTS AND NINJAFOAM)
    set(testdir ${PROJECT_BINARY_DIR}/autotest/config/momentum_backend_unavailable)
    file(MAKE_DIRECTORY ${testdir})
    file(COPY ${PROJECT_SOURCE_DIR}/autotest/config/momentum_backend_unavailable.cfg
              ${PROJECT_SOURCE_DIR}/data/big_butte_small.tif
         DESTINATION ${testdir})
    add_test(NAME cfg_momentum_backend_unavailable
             COMMAND ${PROJECT_BINARY_DIR}/src/cli/WindNinja_cli momentum_backend_unavailable.cfg
             WORKING_DIRECTORY ${testdir})
    set_tests_properties(cfg_momentum_backend_unavailable PROPERTIES
                         ENVIRONMENT "WINDNINJA_DATA=${PROJECT_SOURCE_DIR}/data"
                         PASS_REGULAR_EXPRESSION "'momentum_backend' of webgpu is not valid or not available in this build")
endif()

