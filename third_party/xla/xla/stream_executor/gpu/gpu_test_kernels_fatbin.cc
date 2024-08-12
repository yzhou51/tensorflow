/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/stream_executor/gpu/gpu_test_kernels_fatbin.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tsl/platform/env.h"

namespace stream_executor::gpu {

absl::StatusOr<absl::Span<const uint8_t>> GetGpuTestKernelsFatbin() {
  static absl::StatusOr<std::string>* fatbin = []() {
    tsl::Env* env = tsl::Env::Default();
    auto fatbin = new absl::StatusOr<std::string>{std::string{}};
    absl::Status result =
        tsl::ReadFileToString(env, FATBIN_SRC, &fatbin->value());
    if (!result.ok()) {
      *fatbin = result;
    }
    return fatbin;
  }();

  if (!fatbin->ok()) {
    return fatbin->status();
  }
  return absl::Span<const uint8_t>{
      reinterpret_cast<const uint8_t*>(fatbin->value().data()),
      fatbin->value().size()};
}
}  // namespace stream_executor::gpu
