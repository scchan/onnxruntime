#include "contrib_ops/cpu/activations.h"
#include "core/session/ort_env.h"
#include "core/graph/model.h"
#include "core/graph/graph.h"

#include <random>
#if 0
using namespace onnxruntime;
using namespace onnx;
extern OrtEnv* env;

static float* GenerateFloatArray(size_t batch_size, float low, float high) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(low, high);
	float* data = (float*)_aligned_malloc(sizeof(float) * batch_size, 64);
	for (size_t i = 0; i != batch_size; ++i) {
		data[i] = dist(gen);
	}
	return data;
}


static void CreateKernel(){ 
  auto test_logger = env->GetLoggingManager()->CreateLogger("test");
  Model model("graph_1", false, *test_logger);  
  auto& graph = model.MainGraph();
  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  auto& input_arg = graph.GetOrCreateNodeArg("node_1_in_1", &tensor_int32);
  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &tensor_int32);
  Node& main_node = graph.AddNode("main", "Gelu","",{&input_arg},{&output_arg},nullptr,kMSDomain);
  std::unique_ptr<KernelDef> kernelDef = KernelDefBuilder().SetName("Gelu").SetDomain(kMSDomain).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).Build();

}

#endif