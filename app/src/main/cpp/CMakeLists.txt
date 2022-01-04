cmake_minimum_required(VERSION 3.4.1)

#set(lib_DIR ${CMAKE_SOURCE_DIR}/libs)
#set(inc_DIR ${CMAKE_SOURCE_DIR}/includes)
#include_directories(${inc_DIR})                #指定头文件路径
#include_directories(${inc_DIR}/inc_mnn)                 #指定头文件路径
#include_directories(${inc_DIR}/inc_opencv)             #指定头文件路径

include_directories(includes)

#set(OpenCV_STATIC ON)
#set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/OpenCV-android-sdk/sdk/native/jni)
#find_package(OpenCV REQUIRED)

add_library(
        native-lib
        SHARED

        native-lib.cpp

        imgProcess.cpp
        Bbox.cpp
        mtcnn.cpp
        net.cpp
        face_net.cpp
        face_recognize.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}")

#add_library(MNN SHARED IMPORTED )
#set_target_properties(MNN PROPERTIES IMPORTED_LOCATION ${lib_DIR}/${ANDROID_ABI}/libMNN.so ${lib_DIR}/${ANDROID_ABI}/libMNN_CL.so ${lib_DIR}/${ANDROID_ABI}/libMNN_GL.so ${lib_DIR}/${ANDROID_ABI}/libMNN_Express.so ${lib_DIR}/${ANDROID_ABI}/libMNN_Vulkan.so)

find_library(log-lib log )

target_link_libraries(
        native-lib

        MNN
        MNN_CL
        MNN_GL
        MNN_Express
        MNN_Vulkan

        opencv_java4

        jnigraphics

        android

        ${log-lib}
        ${OpenCV_LIBS} )
