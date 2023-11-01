/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PJRT_C_PJRT_C_API_TEST_H_
#define XLA_PJRT_C_PJRT_C_API_TEST_H_

#include <functional>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace pjrt {

// This struct is used to store the expected outputs for different devices.
struct ExpectedOutputs {
  std::string device_debug_string;
  std::string device_to_string;
};

// Registers a function that generates a PJRT_Api to the test factory. Including
// tensorflow/compiler/xla/pjrt/c/pjrt_c_api_test.h in the test file will run
// all the tests in this test factory with the PJRT_Api generated by the input
// to  RegisterPjRtCApiTestFactory. See
// tensorflow/compiler/xla/pjrt/c/pjrt_c_api_cpu_test.cc for an example usage
void RegisterPjRtCApiTestFactory(
    std::function<const PJRT_Api*()> factory, absl::string_view platform_name,
    std::optional<ExpectedOutputs> expected_outputs);

}  // namespace pjrt

#endif  // XLA_PJRT_C_PJRT_C_API_TEST_H_
