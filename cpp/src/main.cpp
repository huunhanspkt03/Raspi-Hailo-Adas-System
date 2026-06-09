#include <csignal>
#include <iostream>
#include <memory>

#include "adas/cli.hpp"
#include "adas/runtime.hpp"

namespace {
adas::Runtime* g_runtime = nullptr;
void on_signal(int) {
    if (g_runtime) g_runtime->request_stop();
}
}  // namespace

int main(int argc, char** argv) {
    adas::RuntimeConfig cfg;
    std::string err;
    int rc = adas::parse_cli(argc, argv, &cfg, &err);
    if (rc == 2) return 0;       // help printed
    if (rc != 0) {
        std::cerr << "[adas_cpp] error: " << err << "\n\n";
        adas::print_usage(argv[0]);
        return 1;
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    adas::Runtime runtime(std::move(cfg));
    g_runtime = &runtime;

    return runtime.run();
}
