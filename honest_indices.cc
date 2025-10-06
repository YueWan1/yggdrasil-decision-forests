/*
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Honest Forest with Kernel Method training example with In-Bag Indices Export
//
// This program demonstrates:
//   - Training an Honest Random Forest with Kernel Method
//   - Evaluating the model on test dataset
//   - Exporting in-bag sample indices for each tree
//   - Saving the trained model
//
// Usage example:
//   bazel build //examples:train_honest_kernel_forest
//   ./bazel-bin/examples/train_honest_kernel_forest --alsologtostderr

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.pb.h"
#include "yggdrasil_decision_forests/metric/report.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"

// Flags
ABSL_FLAG(std::string, dataset_dir,
          "cv_exports",
          "Input directory containing adult_train.csv and adult_test.csv");

ABSL_FLAG(std::string, output_dir, "/tmp/honest_forest",
          "Output directory to save the model and results");

// Honest Forest 
ABSL_FLAG(bool, enable_honest, true, "Enable honest forest");
ABSL_FLAG(float, honest_ratio, 0.5, "Honest ratio for leaf examples");
ABSL_FLAG(bool, honest_fixed_separation, false, "Fixed separation for honest trees");

// Kernel Method 
ABSL_FLAG(bool, enable_kernel, true, "Whether to use kernel method");

// Random Forest 
ABSL_FLAG(int, num_trees, 1, "Number of trees");
ABSL_FLAG(bool, winner_take_all, false, "Winner take all inference");
ABSL_FLAG(float, bootstrap_ratio, 1.6, "Bootstrap sampling ratio");

// Oblique splits
ABSL_FLAG(int, num_threads, 1, "Number of threads to use.");
ABSL_FLAG(int, tree_depth, -1,
          "Maximum depth of trees (-1 for unlimited).");
ABSL_FLAG(int, max_num_projections, 1000,
          "Maximum number of projections for oblique splits.");
ABSL_FLAG(float, projection_density_factor, 1.5f,
          "Projection density factor.");
ABSL_FLAG(float, num_projections_exponent, 0.5,
          "Exponent to determine number of projections.");
ABSL_FLAG(int, min_examples, 1, "Min examples in splits");

// In-bag indices export
ABSL_FLAG(bool, export_inbag_indices, true, 
          "Export in-bag sample indices for each tree");

namespace ydf = yggdrasil_decision_forests;

int main(int argc, char** argv) {
  // Enable the logging 
  InitLogging(argv[0], &argc, &argv, true);

  // Read flags 
  const std::string dataset_dir = absl::GetFlag(FLAGS_dataset_dir);
  const std::string output_dir = absl::GetFlag(FLAGS_output_dir);

  // Training and testing dataset paths 
  const auto train_path =
      absl::StrCat("csv:", file::JoinPath(dataset_dir, "fold1_train.csv"));
  const auto test_path =
      absl::StrCat("csv:", file::JoinPath(dataset_dir, "fold1_test.csv"));

  QCHECK_OK(file::RecursivelyCreateDir(output_dir, file::Defaults()));

  LOG(INFO) << "===  Honest Setting  w/wo kernel Training ===";
  LOG(INFO) << "Training data: " << train_path;
  LOG(INFO) << "Test data: " << test_path;

  LOG(INFO) << "Create dataspec";
  const auto dataspec_path = file::JoinPath(output_dir, "dataspec.pbtxt");
  
  ydf::dataset::proto::DataSpecificationGuide guide;
  auto* col_guide = guide.add_column_guides();
  col_guide->set_column_name_pattern("label");
  col_guide->set_type(ydf::dataset::proto::ColumnType::CATEGORICAL);
  
  const auto dataspec = ydf::dataset::CreateDataSpec(train_path, guide).value();
  QCHECK_OK(file::SetTextProto(dataspec_path, dataspec, file::Defaults()));

  // Load training dataset to get number of examples
  LOG(INFO) << "Load training dataset";
  ydf::dataset::VerticalDataset train_dataset;
  QCHECK_OK(ydf::dataset::LoadVerticalDataset(train_path, dataspec, &train_dataset));
  const auto num_train_examples = train_dataset.nrow();
  LOG(INFO) << "Training dataset has " << num_train_examples << " examples";
  
  LOG(INFO) << "Configure Kernel Method setting";
  ydf::model::proto::TrainingConfig train_config;
  train_config.set_learner("RANDOM_FOREST");
  train_config.set_task(ydf::model::proto::Task::CLASSIFICATION);
  train_config.set_label("label");

  auto& rf_config = *train_config.MutableExtension(
      ydf::model::random_forest::proto::random_forest_config);
  
  rf_config.set_num_trees(absl::GetFlag(FLAGS_num_trees));
  rf_config.set_winner_take_all_inference(absl::GetFlag(FLAGS_winner_take_all));
  rf_config.set_bootstrap_training_dataset(true);
  rf_config.set_bootstrap_size_ratio(absl::GetFlag(FLAGS_bootstrap_ratio));
  
  if (absl::GetFlag(FLAGS_enable_kernel)) {
    LOG(INFO) << "Enabling Kernel Method";
    rf_config.set_kernel_method(true);
  }
  else {
    LOG(INFO) << "Disabling Kernel Method";
    rf_config.set_kernel_method(false);
  }

  // Sparse oblique setting
  auto* dt_config = rf_config.mutable_decision_tree();
  auto* sos = dt_config->mutable_sparse_oblique_split();
  sos->set_max_num_projections(
      absl::GetFlag(FLAGS_max_num_projections));
  sos->set_projection_density_factor(
      absl::GetFlag(FLAGS_projection_density_factor));
  sos->set_num_projections_exponent(
      absl::GetFlag(FLAGS_num_projections_exponent));

  if (absl::GetFlag(FLAGS_enable_honest)) {
    LOG(INFO) << "Enabling Honest Forest";
    auto* honest_config = dt_config->mutable_honest();
    honest_config->set_ratio_leaf_examples(absl::GetFlag(FLAGS_honest_ratio));
    honest_config->set_fixed_separation(absl::GetFlag(FLAGS_honest_fixed_separation));
    
    LOG(INFO) << "  - Honest ratio: " << absl::GetFlag(FLAGS_honest_ratio);
    LOG(INFO) << "  - Fixed separation: " << absl::GetFlag(FLAGS_honest_fixed_separation);
  }
  else {
    LOG(INFO) << "Disabling Honesty";
  }

  std::unique_ptr<ydf::model::AbstractLearner> learner;
  CHECK_OK(ydf::model::GetLearner(train_config, &learner));

  auto* rf_learner = dynamic_cast<ydf::model::random_forest::RandomForestLearner*>(learner.get());
  if (!rf_learner) {
    LOG(FATAL) << "Failed to cast learner to RandomForestLearner";
  }

  LOG(INFO) << "Train model";
  auto model = rf_learner->TrainWithStatus(train_path, dataspec).value();

  if (absl::GetFlag(FLAGS_export_inbag_indices)) {
    LOG(INFO) << "Exporting in-bag sample indices for each tree";
    const auto inbag_dir = file::JoinPath(output_dir, "inbag_indices");
    QCHECK_OK(file::RecursivelyCreateDir(inbag_dir, file::Defaults()));
    
    for (int tree_idx = 0; tree_idx < absl::GetFlag(FLAGS_num_trees); ++tree_idx) {
      auto inbag_indices_or = rf_learner->GetTrainingExampleIndices(
          num_train_examples, tree_idx);
      
      if (!inbag_indices_or.ok()) {
        LOG(ERROR) << "Failed to get in-bag indices for tree " << tree_idx 
                   << ": " << inbag_indices_or.status();
        continue;
      }
      
      const auto& inbag_indices = inbag_indices_or.value();
      
      const auto indices_path = file::JoinPath(
          inbag_dir, absl::StrCat("tree_", tree_idx, "_inbag.txt"));
      
      std::string indices_content;
      indices_content.reserve(inbag_indices.size() * 10); // Rough estimate
      for (const auto idx : inbag_indices) {
        absl::StrAppend(&indices_content, idx, "\n");
      }
      
      QCHECK_OK(file::SetContent(indices_path, indices_content));
      
      if (tree_idx % 100 == 0) {
        LOG(INFO) << "Exported in-bag indices for tree " << tree_idx 
                  << " (" << inbag_indices.size() << " samples)";
      }
    }
    
    LOG(INFO) << "In-bag indices exported to " << inbag_dir;
  }

  LOG(INFO) << "Export the model";
  const auto model_path = file::JoinPath(output_dir, "model");
  QCHECK_OK(ydf::model::SaveModel(model_path, *model));


  LOG(INFO) << "Evaluate model";
  ydf::dataset::VerticalDataset test_dataset;
  QCHECK_OK(ydf::dataset::LoadVerticalDataset(test_path, model->data_spec(),
                                              &test_dataset));

  ydf::utils::RandomEngine rnd;
  const auto evaluation = model->Evaluate(test_dataset, {}, &rnd);

  std::string evaluation_path = file::JoinPath(output_dir, "evaluation.pbtxt");
  QCHECK_OK(file::SetTextProto(evaluation_path, evaluation, file::Defaults()));

  std::string evaluation_report = ydf::metric::TextReport(evaluation).value();
  QCHECK_OK(file::SetContent(absl::StrCat(evaluation_path, ".txt"),
                             evaluation_report));
  LOG(INFO) << "Evaluation:\n" << evaluation_report;

  LOG(INFO) << " Honesty  w/wo Kernel Method completed";
  LOG(INFO) << "The results are available in " << output_dir;

  return 0;
}