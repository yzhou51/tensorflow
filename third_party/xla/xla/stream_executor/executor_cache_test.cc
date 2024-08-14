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

#include "xla/stream_executor/executor_cache.h"

#include <memory>
#include <vector>

#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor {
namespace {

TEST(ExecutorCacheTest, GetOnEmptyCacheFails) {
  ExecutorCache cache;
  EXPECT_FALSE(cache.Get(0).ok());
}

TEST(ExecutorCacheTest, GetOrCreateConstructsAndRepeatedlyReturns) {
  ExecutorCache cache;
  StreamExecutor *created = nullptr;
  auto factory = [&created]() {
    static bool called = false;
    EXPECT_FALSE(called);
    called = true;
    auto executor = std::make_unique<MockStreamExecutor>();
    created = executor.get();
    return executor;
  };
  TF_ASSERT_OK_AND_ASSIGN(auto executor, cache.GetOrCreate(0, factory));
  EXPECT_EQ(executor, created);
  TF_ASSERT_OK_AND_ASSIGN(auto found, cache.GetOrCreate(0, factory));
  EXPECT_EQ(found, created);
  TF_ASSERT_OK_AND_ASSIGN(found, cache.Get(0));
  EXPECT_EQ(found, created);
}

}  // namespace
}  // namespace stream_executor
