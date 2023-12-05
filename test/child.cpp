#include <iostream>
#include <array>
#include <string>
#include <vector>
#include <Windows.h>

#include <migraphx/ranges.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/msgpack.hpp>

void read_stdin()
{
    std::vector<char> result;
    constexpr std::size_t BUFFER_SIZE = 1024;
    DWORD bytes_read;
    TCHAR buffer[BUFFER_SIZE];

    HANDLE std_in  = GetStdHandle(STD_INPUT_HANDLE);
    HANDLE std_out = GetStdHandle(STD_OUTPUT_HANDLE);

    if(std_in == INVALID_HANDLE_VALUE)
        MIGRAPHX_THROW("STDIN invalid handle (" + std::to_string(GetLastError()) + ")");
    if(std_out == INVALID_HANDLE_VALUE)
        MIGRAPHX_THROW("STDOUT invalid handle (" + std::to_string(GetLastError()) + ")");
    
    for(;;)
    {
        BOOL status = ReadFile(std_in, buffer, BUFFER_SIZE, &bytes_read, nullptr);
        if(status == FALSE or bytes_read == 0)
            break;
        DWORD written;
        if(WriteFile(std_out, buffer, bytes_read, &written, nullptr) == FALSE)
            break;
    }        
}

int main(int argc, char const* argv[]) {    
    read_stdin(); 
}
