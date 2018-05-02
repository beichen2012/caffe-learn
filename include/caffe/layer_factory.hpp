/**
 * @brief A layer factory that allows one to register layers. layer��������ע���
 * During runtime, registered layers can be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:  �������ڼ䣬ע��Ĳ����ͨ���� CreateLayer���� protobuffer���͵�layer param������ 
 *
 *     LayerRegistry<Dtype>::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:���������ַ�ʽע��һ���㣬�ٶ���һ���㣺
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> { //���м̳��� Layer
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Layer" at the end  �������;�����������������û��Layer�⼸����ĸ������
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line: �����������ù��캯���򵥹���ģ���ôֻ����cpp�ļ������һ�䣺
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of: ���������Ǳ������Ĺ��캯������ģ����磺
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like �����������ж��backends���ο�GetConvolutionLayer����
 *�����Ҫ����ע�᣺ 
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.   ÿһ�ֲ�����ֻ����ע��һ��
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
  typedef std::map<string, Creator> CreatorRegistry;		//�����������㹹�캯����map

  static CreatorRegistry& Registry();					//��̬���й��캯��

  // Adds a creator.
  static void AddCreator(const string& type, Creator creator);

  // Get a layer using a LayerParameter. ʹ��param�����type����������һ��layer, �����Զ�����Ժ�...
  static shared_ptr<Layer<Dtype> > CreateLayer(const LayerParameter& param);

  static vector<string> LayerTypeList(); //�����Ѿ�ע��Ĳ���б�

 private:
  // Layer registry should never be instantiated - everything is done with its
  // static variables.
  LayerRegistry();

  static string LayerTypeListString(); //�������Ѿ�ע��Ĳ㣬���س�һ���ַ������м��ö��Ÿ���
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
