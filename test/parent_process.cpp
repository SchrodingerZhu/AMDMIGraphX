#include <Windows.h>
#include <iostream>
#include <string>

#include "test.hpp"
#include <migraphx/msgpack.hpp>
#include <migraphx/process.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/errors.hpp>
#include <tchar.h>

#define BUFSIZE MAX_PATH
    
std::string get_cwd()
{
    char Buffer[BUFSIZE];
    DWORD dwRet;
    dwRet = GetCurrentDirectory(BUFSIZE, Buffer);
    if(dwRet == 0)
        MIGRAPHX_THROW("GetCurrentDirectory failed (" + std::to_string(GetLastError()) + ")");
    return std::string(Buffer);
}

TEST_CASE(string_data)
{
    std::string cwd = get_cwd();
    auto child_path = migraphx::fs::path{cwd};

    std::string string_data = "Parent string";

    // write string data to child process
    migraphx::process{"test_child.exe"}.cwd(child_path).write([&](auto writer) {
        migraphx::to_msgpack(string_data, writer);
    });

    //// parent process read from child stdout
    // std::vector<char> result;
    // HANDLE std_in = GetStdHandle(STD_INPUT_HANDLE);
    // if(std_in == INVALID_HANDLE_VALUE)
    //     MIGRAPHX_THROW("STDIN invalid handle (" + std::to_string(GetLastError()) + ")");
    // constexpr std::size_t BUFFER_SIZE = 4096;
    // DWORD bytes_read;
    // TCHAR buffer[BUFFER_SIZE];
    // for(;;)
    // {
    //     BOOL status = ReadFile(std_in, buffer, BUFFER_SIZE, &bytes_read, nullptr);
    //     if(status == FALSE or bytes_read == 0)
    //         break;

    //    result.insert(result.end(), buffer, buffer + bytes_read);
    // }

    // EXPECT(result.data() == string_data);
}

TEST_CASE(binary_data)
{

    // binary data
    std::vector<char> binary_data = {'B', 'i', 'n', 'a', 'r', 'y'};
    std::string cwd               = get_cwd();
    auto child_path               = migraphx::fs::path{cwd};

    // write string data to child process
    migraphx::process{"test_child.exe"}.cwd(child_path).write([&](auto writer) {
        migraphx::to_msgpack(binary_data, writer);
    });

    //// parent process read from child stdout
    // std::vector<char> result;
    // HANDLE std_in = GetStdHandle(STD_INPUT_HANDLE);
    // if(std_in == INVALID_HANDLE_VALUE)
    //     MIGRAPHX_THROW("STDIN invalid handle (" + std::to_string(GetLastError()) + ")");
    // constexpr std::size_t BUFFER_SIZE = 4096;
    // DWORD bytes_read;
    // TCHAR buffer[BUFFER_SIZE];
    // for(;;)
    //{
    //     BOOL status = ReadFile(std_in, buffer, BUFFER_SIZE, &bytes_read, nullptr);
    //     if(status == FALSE or bytes_read == 0)
    //         break;

    //    result.insert(result.end(), buffer, buffer + bytes_read);
    //}

    // EXPECT(result.data() == string_data);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
