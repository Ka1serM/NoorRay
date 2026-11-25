#include "NoorRay.OpenDlssRr/Weights.h"

#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <format>
#include <fstream>
#include <mutex>
#include <thread>

#include <OpenDlssRr/RayReconstruction.h>

#include "NoorRay.OpenDlssRr/Http.h"
#include "NoorRay/Core/Log.h"

namespace NoorRay::OpenDlssRr {

namespace {

enum class State : std::uint8_t { Idle, Downloading, Ready, Failed };

struct Download {
    std::mutex mutex;
    std::condition_variable finished;
    State state = State::Idle;
    std::string error;
};

//! Never freed, because a detached download thread can still run while the program exits.
Download& getDownload() {
    static auto* download = new Download;
    return *download;
}

void finish(Download& download, const State state, std::string error = {}) {
    {
        const std::scoped_lock lock(download.mutex);
        download.state = state;
        download.error = std::move(error);
    }
    download.finished.notify_all();
}

constexpr const char* rangeName = "libnvidia-ngx-dlssd.so.310.9.16.weights";

std::expected<void, Core::Error> fetch() {
    const ::OpenDlssRr::LibraryRange& range = ::OpenDlssRr::libraryRange;
    const std::filesystem::path folder = ::OpenDlssRr::getLibraryFolder();
    if (folder.empty())
        return Core::fail(Core::ErrorCode::Io, "OpenDLSS-RR has no user folder for the weights");
    Core::Log::info("downloading the OpenDLSS-RR weights ({:.1f} MB) into {}", static_cast<double>(range.size) / 1e6, folder.string());
    const auto start = std::chrono::steady_clock::now();
    std::expected<std::vector<std::byte>, Core::Error> bytes = downloadRange(range.url, range.offset, range.size);
    if (!bytes)
        return std::unexpected(std::move(bytes.error()));
    if (auto checked = ::OpenDlssRr::checkLibrary(*bytes); !checked)
        return Core::fail(Core::ErrorCode::InvalidFile, "the downloaded OpenDLSS-RR weights are wrong: " + checked.error().message);
    std::error_code error;
    std::filesystem::create_directories(folder, error);
    //! Another NoorRay process may search the folder during the write, so the file appears only by the rename.
    const std::filesystem::path target = folder / rangeName;
    const std::filesystem::path partial = folder / std::format("{}.{}.part", rangeName, std::hash<std::thread::id>{}(std::this_thread::get_id()));
    {
        std::ofstream file(partial, std::ios::binary | std::ios::trunc);
        if (!file.write(reinterpret_cast<const char*>(bytes->data()), static_cast<std::streamsize>(bytes->size())))
            return Core::fail(Core::ErrorCode::Io, "cannot write " + partial.string());
    }
    std::filesystem::rename(partial, target, error);
    if (error) {
        std::filesystem::remove(partial, error);
        return Core::fail(Core::ErrorCode::Io, "cannot write " + target.string());
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    Core::Log::info("downloaded the OpenDLSS-RR weights in {:.1f} s: {}", seconds, target.string());
    return {};
}

}

void Weights::prefetch() {
    Download& download = getDownload();
    {
        const std::scoped_lock lock(download.mutex);
        if (download.state != State::Idle)
            return;
        download.state = State::Downloading;
    }
    if (::OpenDlssRr::findLibrary()) {
        finish(download, State::Ready);
        return;
    }
    std::thread([&download] {
        std::expected<void, Core::Error> fetched = fetch();
        if (!fetched)
            Core::Log::warning("OpenDLSS-RR: {}", fetched.error().message);
        finish(download, fetched ? State::Ready : State::Failed, fetched ? std::string() : fetched.error().message);
    }).detach();
}

std::expected<void, Core::Error> Weights::wait() {
    prefetch();
    Download& download = getDownload();
    std::unique_lock lock(download.mutex);
    download.finished.wait(lock, [&download] { return download.state != State::Downloading; });
    if (download.state == State::Failed)
        return Core::fail(Core::ErrorCode::Io, download.error);
    return {};
}

}
