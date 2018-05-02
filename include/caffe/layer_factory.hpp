/**
 * @brief A layer factory that allows one to register layers. layer工厂允许注册层
 * During runtime, registered layers can be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:  在运行期间，注册的层可以通过向 CreateLayer传入 protobuffer类型的layer param来调用 
 *
 *     LayerRegistry<Dtype>::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:这里有两种方式注册一个层，假定有一个层：
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> { //仅有继承自 Layer
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Layer" at the end  它的类型就是它的类名，但是没有Layer这几个字母，即：
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line: 如果这个类是用构造函数简单构造的，那么只用在cpp文件里，加上一句：
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of: 如果这个类是被其他的构造函数构造的，例如：
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like （如果你的类有多个backends，参考GetConvolutionLayer），
 *你就需要这样注册： 
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.   每一种层类型只允许被注册一次
 */

#ifndef CAFFE_LAYER_FACTORY_H_
#define CAFFE_LAYER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Layer;

template <typename Dtype>
class LayerRegistry {
 public:
  typedef shared_ptr<Layer<Dtype> > (*Creator)(const LayerParameter&);
  typedef std::map<string, Creator> CreatorRegistry;		//保存层名称与层构造函数的map

  static CreatorRegistry& Registry();					//静态单列构造函数

  // Adds a creator.
  static void AddCreator(const string& type, Creator creator);

  // Get a layer using a LayerParameter. 使用param里面的type参数，构造一个layer, 所以自定义层以后...
  static shared_ptr<Layer<Dtype> > CreateLayer(const LayerParameter& param);

  static vector<string> LayerTypeList(); //返回已经注册的层的列表

 private:
  // Layer registry should never be instantiated - everything is done with its
  // static variables.
  LayerRegistry();

  static string LayerTypeListString(); //把所有已经注册的层，返回成一个字符串，中间用逗号隔开
};

template <typename Dtype>
class LayerRegisterer {
 public:
  LayerRegisterer(const string& type,
                  shared_ptr<Layer<Dtype> > (*creator)(const LayerParameter&));
};

#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \

#define REGISTER_LAYER_CLASS(type)                                             \
  template <typename Dtype>                                                    \
  shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param) \
  {                                                                            \
    return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));           \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_H_
