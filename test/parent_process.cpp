#include <Windows.h>
#include <tchar.h>
#include <iostream>
#include <cstring>

#include "test.hpp"
#include <migraphx/errors.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/msgpack.hpp>
#include <migraphx/process.hpp>

#define BUFSIZE MAX_PATH
constexpr std::size_t BUFFER_SIZE = 4096;
STARTUPINFO info;
PROCESS_INFORMATION process_info;

enum class direction
{
    input,
    output
};

template <direction dir>
class pipe
{
    public:
    explicit pipe()
    {
        SECURITY_ATTRIBUTES attrs;
        attrs.nLength              = sizeof(SECURITY_ATTRIBUTES);
        attrs.bInheritHandle       = TRUE;
        attrs.lpSecurityDescriptor = nullptr;

        if(CreatePipe(&m_read, &m_write, &attrs, 0) == FALSE)
            throw GetLastError();

        if(dir == direction::output)
        {
            // Do not inherit the read handle for the output pipe
            if(SetHandleInformation(m_read, HANDLE_FLAG_INHERIT, 0) == 0)
                throw GetLastError();
        }
        else
        {
            // Do not inherit the write handle for the input pipe
            if(SetHandleInformation(m_write, HANDLE_FLAG_INHERIT, 0) == 0)
                throw GetLastError();
        }
    }

    pipe(const pipe&)            = delete;
    pipe& operator=(const pipe&) = delete;

    pipe(pipe&&) = default;

    ~pipe()
    {
        if(m_write != nullptr)
        {
            CloseHandle(m_write);
        }
        if(m_read != nullptr)
        {
            CloseHandle(m_read);
        }
    }

    bool close_write_handle()
    {
        auto result = true;
        if(m_write != nullptr)
        {
            result  = CloseHandle(m_write) == TRUE;
            m_write = nullptr;
        }
        return result;
    }

    bool close_read_handle()
    {
        auto result = true;
        if(m_read != nullptr)
        {
            result = CloseHandle(m_read) == TRUE;
            m_read = nullptr;
        }
        return result;
    }

    std::pair<bool, DWORD> read(LPVOID buffer, DWORD length) const
    {
        DWORD bytes_read;
        if(ReadFile(m_read, buffer, length, &bytes_read, nullptr) == FALSE and
            GetLastError() == ERROR_MORE_DATA)
        {
            return {true, bytes_read};
        }
        return {false, bytes_read};
    }

    HANDLE get_read_handle() const { return m_read; }

    bool write(LPCVOID buffer, DWORD length) const
    {
        DWORD bytes_written;
        return WriteFile(m_write, buffer, length, &bytes_written, nullptr) == TRUE;
    }

    HANDLE get_write_handle() const { return m_write; }

        
    HANDLE m_write = nullptr, m_read = nullptr;
};

std::string get_cwd()
{
    char Buffer[BUFSIZE];
    DWORD dwRet;
    dwRet = GetCurrentDirectory(BUFSIZE, Buffer);
    if(dwRet == 0)
        MIGRAPHX_THROW("GetCurrentDirectory failed (" + std::to_string(GetLastError()) + ")");
    return std::string(Buffer);
}

void CreateProcess(pipe<direction::input>& input,
                pipe<direction::output>& output,
                const std::string& child_process_name,
                const std::string& cwd)
{
    ZeroMemory(&info, sizeof(STARTUPINFO));
    info.cb         = sizeof(STARTUPINFO);
    info.hStdError  = output.get_write_handle();
    info.hStdOutput = output.get_write_handle();
    info.hStdInput  = input.get_read_handle();
    info.dwFlags |= STARTF_USESTDHANDLES;

    TCHAR cmdline[MAX_PATH];
    std::strncpy(cmdline, child_process_name.c_str(), MAX_PATH);

    ZeroMemory(&process_info, sizeof(process_info));

    if(CreateProcess(nullptr,
                        cmdline,
                        nullptr,
                        nullptr,
                        TRUE,
                        0,
                        nullptr,
                        cwd.empty() ? nullptr : static_cast<LPCSTR>(cwd.c_str()),
                        &info,
                        &process_info) == FALSE)
    {
        MIGRAPHX_THROW("Error creating process (" + std::to_string(GetLastError()) + ")");
    }

    if(not output.close_write_handle())
        MIGRAPHX_THROW("Error closing STDOUT handle for writing (" +
                        std::to_string(GetLastError()) + ")");

    if(not input.close_read_handle())
        MIGRAPHX_THROW("Error closing STDIN handle for reading (" +
                        std::to_string(GetLastError()) + ")");
}

void write_to_child(LPCVOID buffer, std::size_t n, pipe<direction::input>& input)
{
    DWORD bytes_written;
    if(WriteFile(input.m_write, buffer, n, &bytes_written, nullptr) ==
    FALSE) 
    {
        MIGRAPHX_THROW("Error writing to child STDIN (" + std::to_string(GetLastError()) + ")");
    }

    if(not input.close_write_handle())
        MIGRAPHX_THROW("Error closing STDIN handle for writing (" + std::to_string(GetLastError()) +
                        ")");
}

std::vector<char> read_from_child(pipe<direction::output>& output) {
    std::vector<char> result;
    DWORD bytes_read;
    TCHAR buffer[BUFFER_SIZE];

    for(;;)
    {
        BOOL status = ReadFile(output.m_read, buffer, BUFFER_SIZE, &bytes_read, nullptr);
        if(status == FALSE or bytes_read == 0)
            break;

        result.insert(result.end(), buffer, buffer + bytes_read);
    }
    return result;
}


TEST_CASE(string_data)
{
    std::string cwd = get_cwd();

    std::string string_data = "Parent string";

    std::string child_process_name = "test_child.exe";

    pipe<direction::input> input{};
    pipe<direction::output> output{};

    CreateProcess(input, output, child_process_name, cwd);

    // write to child process
    TCHAR buffer[BUFFER_SIZE];
    std::strncpy(buffer, string_data.c_str(), BUFFER_SIZE);  
    write_to_child(buffer, BUFFER_SIZE, input);

    // read from child stdout
    std::vector<char> result = read_from_child(output);
        
    WaitForSingleObject(process_info.hProcess, INFINITE);

    DWORD status{};
    GetExitCodeProcess(process_info.hProcess, &status);

    CloseHandle(process_info.hProcess);
    CloseHandle(process_info.hThread);
    
    //compare input parent and output child process
    EXPECT(result.data() == string_data);
}

TEST_CASE(binary_data)
{
    std::string cwd = get_cwd();

    std::vector<char> binary_data = {'B', 'i', 'n', 'a', 'r', 'y'};

    std::string child_process_name = "test_child.exe";

    pipe<direction::input> input{};
    pipe<direction::output> output{};

    CreateProcess(input, output, child_process_name, cwd);

    write_to_child(binary_data.data(), binary_data.size(), input);

    // read from child stdout
    std::vector<char> result = read_from_child(output);

    WaitForSingleObject(process_info.hProcess, INFINITE);

    DWORD status{};
    GetExitCodeProcess(process_info.hProcess, &status);

    CloseHandle(process_info.hProcess);
    CloseHandle(process_info.hThread);

    // compare input parent and output child process
    EXPECT(result == binary_data);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
