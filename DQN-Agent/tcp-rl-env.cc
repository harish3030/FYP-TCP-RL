#include "tcp-rl-env.h"
#include "ns3/tcp-header.h"
#include "ns3/object.h"
#include "ns3/core-module.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/tcp-socket-base.h"
#include <vector>
#include <numeric>
#include <cmath>
namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ns3::TcpGymEnv");
NS_OBJECT_ENSURE_REGISTERED (TcpGymEnv);

TcpGymEnv::TcpGymEnv ()
{
  NS_LOG_FUNCTION (this);
  SetOpenGymInterface(OpenGymInterface::Get());
}

TcpGymEnv::~TcpGymEnv ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
TcpGymEnv::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpGymEnv")
    .SetParent<OpenGymEnv> ()
    .SetGroupName ("OpenGym")
  ;

  return tid;
}

void
TcpGymEnv::DoDispose ()
{
  // NS_LOG_FUNCTION (this);
}

void
TcpGymEnv::SetNodeId(uint32_t id)
{
  // NS_LOG_FUNCTION (this);
  m_nodeId = id;
}

void
TcpGymEnv::SetSocketUuid(uint32_t id)
{
  // NS_LOG_FUNCTION (this);
  m_socketUuid = id;
}

std::string
TcpGymEnv::GetTcpCongStateName(const TcpSocketState::TcpCongState_t state)
{
  std::string stateName = "UNKNOWN";
  switch(state) {
    case TcpSocketState::CA_OPEN:
      stateName = "CA_OPEN";
      break;
    case TcpSocketState::CA_DISORDER:
      stateName = "CA_DISORDER";
      break;
    case TcpSocketState::CA_CWR:
      stateName = "CA_CWR";
      break;
    case TcpSocketState::CA_RECOVERY:
      stateName = "CA_RECOVERY";
      break;
    case TcpSocketState::CA_LOSS:
      stateName = "CA_LOSS";
      break;
    case TcpSocketState::CA_LAST_STATE:
      stateName = "CA_LAST_STATE";
      break;
    default:
       stateName = "UNKNOWN";
       break;
  }
  return stateName;
}

std::string
TcpGymEnv::GetTcpCAEventName(const TcpSocketState::TcpCAEvent_t event)
{
  std::string eventName = "UNKNOWN";
  switch(event) {
    case TcpSocketState::CA_EVENT_TX_START:
      eventName = "CA_EVENT_TX_START";
      break;
    case TcpSocketState::CA_EVENT_CWND_RESTART:
      eventName = "CA_EVENT_CWND_RESTART";
      break;
    case TcpSocketState::CA_EVENT_COMPLETE_CWR:
      eventName = "CA_EVENT_COMPLETE_CWR";
      break;
    case TcpSocketState::CA_EVENT_LOSS:
      eventName = "CA_EVENT_LOSS";
      break;
    case TcpSocketState::CA_EVENT_ECN_NO_CE:
      eventName = "CA_EVENT_ECN_NO_CE";
      break;
    case TcpSocketState::CA_EVENT_ECN_IS_CE:
      eventName = "CA_EVENT_ECN_IS_CE";
      break;
    case TcpSocketState::CA_EVENT_DELAYED_ACK:
      eventName = "CA_EVENT_DELAYED_ACK";
      break;
    case TcpSocketState::CA_EVENT_NON_DELAYED_ACK:
      eventName = "CA_EVENT_NON_DELAYED_ACK";
      break;
    default:
       eventName = "UNKNOWN";
       break;
  }
  return eventName;
}

/*
Define action space
*/
Ptr<OpenGymSpace>
TcpGymEnv::GetActionSpace()
{
  // new_ssThresh
  // new_cWnd
  uint32_t parameterNum = 2;
  float low = 0.0;
  float high = 65535;
  std::vector<uint32_t> shape = {parameterNum,};
  std::string dtype = TypeNameGet<uint32_t> ();

  Ptr<OpenGymBoxSpace> box = CreateObject<OpenGymBoxSpace> (low, high, shape, dtype);
  // NS_LOG_INFO ("MyGetActionSpace: " << box);
  return box;
}

/*
Define game over condition
*/
bool
TcpGymEnv::GetGameOver()
{
  m_isGameOver = false;
  bool test = false;
  static float stepCounter = 0.0;
  stepCounter += 1;
  if (stepCounter == 10 && test) {
      m_isGameOver = true;
  }
  // NS_LOG_INFO ("MyGetGameOver: " << m_isGameOver);
  return m_isGameOver;
}

/*
Define reward function
*/
// float
// TcpGymEnv::GetReward()
// {
  
//   if (TcpTimeStepGymEnv *child = dynamic_cast<TcpTimeStepGymEnv*>(this)) {
//        // child->childMember = 0;
//         if(child->m_tcb!=nullptr){
  
//   //uint64_t diff=m_current_rtt.GetMicroSeconds();
//         uint64_t diff=(child->m_current_rtt-child->m_minRtt).GetMicroSeconds();
//         uint64_t segmentsAckedSum = std::accumulate(child->m_segmentsAcked.begin(), child->m_segmentsAcked.end(), 0);
        
//         float m_throughput = (segmentsAckedSum * child->m_tcb->m_segmentSize) / child->m_timeStep.GetSeconds();
//         std::cout<<"Throughput: "<<m_throughput<<"\n";
//         std::cout<<"Min RTT: "<<child->m_minRtt<<"\n";
//         std::cout<<"Current RTT: "<<child->m_current_rtt<<"\n";
      
        
//         float utility_value=log((m_throughput*8)/(2*1e6))-log(diff);
//         if(utility_value==INFINITY){
//           utility_value=log((m_throughput*8)/(2*1e6));
//         }
//         std::cout<<"Utility value: "<<utility_value<<"\n";
//         std::cout<<"-----------------\n";
        
          
//           //float last_utility_val=m_utilities.back();
//           float delta=utility_value-child->old_U;
//           if(delta>=0)child->m_reward=10;
//           if(delta>=0 and delta<1)child->m_reward=2;
//           if(delta>=-1 and delta<0)child->m_reward=-2;

//           if(delta<-1)child->m_reward=-10;;
          

        
//           child->old_U=utility_value;
//            m_envReward=child->m_reward;
//   }
//   }
 
//   NS_LOG_INFO("MyGetReward: " << m_envReward);

//   return m_envReward;
// }

/*
Define extra info. Optional
*/
std::string
TcpGymEnv::GetExtraInfo()
{
  // NS_LOG_INFO("MyGetExtraInfo: " << m_info);

  return m_info;
}

/*
Execute received actions
*/
bool
TcpGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
  Ptr<OpenGymBoxContainer<uint32_t> > box = DynamicCast<OpenGymBoxContainer<uint32_t> >
                                                                            (action);
  m_new_ssThresh = box->GetValue(0);
  m_new_cWnd = box->GetValue(1);

  NS_LOG_INFO ("MyExecuteActions: " << action);
  return true;
}


NS_OBJECT_ENSURE_REGISTERED (TcpEventGymEnv);

TcpEventGymEnv::TcpEventGymEnv () : TcpGymEnv()
{
  // NS_LOG_FUNCTION (this);
}

TcpEventGymEnv::~TcpEventGymEnv ()
{
  // NS_LOG_FUNCTION (this);
}

TypeId
TcpEventGymEnv::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpEventGymEnv")
    .SetParent<TcpGymEnv> ()
    .SetGroupName ("OpenGym")
    .AddConstructor<TcpEventGymEnv> ()
  ;

  return tid;
}

void
TcpEventGymEnv::DoDispose ()
{
  // NS_LOG_FUNCTION (this);
}
float
TcpEventGymEnv::GetReward(){
  return m_reward;
}
void
TcpEventGymEnv::SetReward(float value)
{
  // NS_LOG_FUNCTION (this);
  m_reward = value;
}

void
TcpEventGymEnv::SetPenalty(float value)
{
  // NS_LOG_FUNCTION (this);
  m_penalty = value;
}

/*
Define observation space
*/
Ptr<OpenGymSpace>
TcpEventGymEnv::GetObservationSpace()
{
  // socket unique ID
  // tcp env type: event-based = 0 / time-based = 1
  // sim time in us
  // node ID
  // ssThresh
  // cWnd
  // segmentSize
  // segmentsAcked
  // bytesInFlight
  // rtt in us
  // min rtt in us
  // called func
  // congestion algorithm (CA) state
  // CA event
  // ECN state
  uint32_t parameterNum = 10;
  float low = 0.0;
  float high = 1000000000.0;
  std::vector<uint32_t> shape = {parameterNum,};
  std::string dtype = TypeNameGet<uint64_t> ();

  Ptr<OpenGymBoxSpace> box = CreateObject<OpenGymBoxSpace> (low, high, shape, dtype);
  // NS_LOG_INFO ("MyGetObservationSpace: " << box);
  return box;
}

/*
Collect observations
*/
Ptr<OpenGymDataContainer>
TcpEventGymEnv::GetObservation()
{
  uint32_t parameterNum = 10;
  std::vector<uint32_t> shape = {parameterNum,};

  Ptr<OpenGymBoxContainer<uint64_t> > box = CreateObject<OpenGymBoxContainer<uint64_t> >(shape);

  box->AddValue(m_socketUuid);
  box->AddValue(0);
  box->AddValue(Simulator::Now().GetMicroSeconds ());
  box->AddValue(m_nodeId);
  box->AddValue(m_tcb->m_ssThresh);
  box->AddValue(m_tcb->m_cWnd);
  box->AddValue(m_tcb->m_segmentSize);
  box->AddValue(m_segmentsAcked);
  box->AddValue(m_bytesInFlight);
  box->AddValue(m_rtt.GetMicroSeconds ());
  //box->AddValue(m_tcb->m_minRtt.GetMicroSeconds ());
  //box->AddValue(m_calledFunc);
  //box->AddValue(m_tcb->m_congState);
  //box->AddValue(m_event);
  //box->AddValue(m_tcb->m_ecnState);

  // Print data
  NS_LOG_INFO ("MyGetObservation: " << box);
  return box;
}

void
TcpEventGymEnv::TxPktTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>)
{
  // NS_LOG_FUNCTION (this);
}

void
TcpEventGymEnv::RxPktTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>)
{
  // NS_LOG_FUNCTION (this);
}

uint32_t
TcpEventGymEnv::GetSsThresh (Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight)
{
  // NS_LOG_FUNCTION (this);
  // pkt was lost, so penalty
  m_envReward = m_penalty;

  NS_LOG_INFO(Simulator::Now() << " Node: " << m_nodeId << " GetSsThresh, BytesInFlight: " << bytesInFlight);
  m_calledFunc = CalledFunc_t::GET_SS_THRESH;
  m_info = "GetSsThresh";
  m_tcb = tcb;
  m_bytesInFlight = bytesInFlight;
  Notify();
  return m_new_ssThresh;
}

void
TcpEventGymEnv::IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
  NS_LOG_FUNCTION (this);
  // pkt was acked, so reward
  m_envReward = m_reward;

  NS_LOG_INFO(Simulator::Now() << " Node: " << m_nodeId << " IncreaseWindow, SegmentsAcked: " << segmentsAcked);
  m_calledFunc = CalledFunc_t::INCREASE_WINDOW;
  m_info = "IncreaseWindow";
  m_tcb = tcb;
  m_segmentsAcked = segmentsAcked;
  Notify();
  tcb->m_cWnd = m_new_cWnd;
}

void
TcpEventGymEnv::PktsAcked (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_INFO(Simulator::Now() << " Node: " << m_nodeId << " PktsAcked, SegmentsAcked: " << segmentsAcked << " Rtt: " << rtt);
  m_calledFunc = CalledFunc_t::PKTS_ACKED;
  m_info = "PktsAcked";
  m_tcb = tcb;
  m_segmentsAcked = segmentsAcked;
  m_rtt = rtt;
}

void
TcpEventGymEnv::CongestionStateSet (Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCongState_t newState)
{
  NS_LOG_FUNCTION (this);
  std::string stateName = GetTcpCongStateName(newState);
  NS_LOG_INFO(Simulator::Now() << " Node: " << m_nodeId << " CongestionStateSet: " << newState << " " << stateName);

  m_calledFunc = CalledFunc_t::CONGESTION_STATE_SET;
  m_info = "CongestionStateSet";
  m_tcb = tcb;
  m_newState = newState;
}

void
TcpEventGymEnv::CwndEvent (Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event)
{
  NS_LOG_FUNCTION (this);
  std::string eventName = GetTcpCAEventName(event);
  NS_LOG_INFO(Simulator::Now() << " Node: " << m_nodeId << " CwndEvent: " << event << " " << eventName);

  m_calledFunc = CalledFunc_t::CWND_EVENT;
  m_info = "CwndEvent";
  m_tcb = tcb;
  m_event = event;
}


NS_OBJECT_ENSURE_REGISTERED (TcpTimeStepGymEnv);

TcpTimeStepGymEnv::TcpTimeStepGymEnv () : TcpGymEnv()
{
  // NS_LOG_FUNCTION (this);
}

void
TcpTimeStepGymEnv::ScheduleNextStateRead ()
{
  // NS_LOG_FUNCTION (this);
  Simulator::Schedule (m_timeStep, &TcpTimeStepGymEnv::ScheduleNextStateRead, this);
  Notify();
}

TcpTimeStepGymEnv::~TcpTimeStepGymEnv ()
{
  // NS_LOG_FUNCTION (this);
}

TypeId
TcpTimeStepGymEnv::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpTimeStepGymEnv")
    .SetParent<TcpGymEnv> ()
    .SetGroupName ("OpenGym")
    .AddConstructor<TcpTimeStepGymEnv> ()
  ;

  return tid;
}

void
TcpTimeStepGymEnv::DoDispose ()
{
  // NS_LOG_FUNCTION (this);
}

void
TcpTimeStepGymEnv::SetDuration(Time value)
{
  // NS_LOG_FUNCTION (this);
  m_duration = value;
}

void
TcpTimeStepGymEnv::SetTimeStep(Time value)
{
  // NS_LOG_FUNCTION (this);
  m_timeStep = value;
}
float
TcpTimeStepGymEnv::GetReward(){
 
//  if(m_tcb!=nullptr){
  
//   uint64_t diff=(m_current_rtt-m_minRtt).GetMicroSeconds();
//   uint64_t segmentsAckedSum = std::accumulate(m_segmentsAcked.begin(), m_segmentsAcked.end(), 
//                                              0);
//   float m_throughput = (segmentsAckedSum * m_tcb->m_segmentSize) / m_timeStep.GetSeconds();
//   float utility_value=log((m_throughput*8)/(10*1e6))-0.01*log(diff);
//   if(std::isinf(utility_value) or std::isnan(utility_value)){
//     utility_value=log((m_throughput*8)/(10*1e6));
//   }
//   float delta=old_U-utility_value;
//   if(delta>=0)m_reward=10;
//   if(delta>=0 and delta<1)m_reward=5;
//   if(delta>=-1 and delta<0)m_reward=-5;
//   if(delta<-1)m_reward=-10;
//   old_U=utility_value;
//  }
  // NS_LOG_INFO("MyGetReward: " << m_reward);
  return m_reward;
}
void
TcpTimeStepGymEnv::SetReward(float value)
{
 
  
  // NS_LOG_FUNCTION (this);
  //m_reward = value;
}

void
TcpTimeStepGymEnv::SetPenalty(float value)
{
  // NS_LOG_FUNCTION (this);
  m_penalty = value;
}



//Define observation space
Ptr<OpenGymSpace> TcpTimeStepGymEnv::GetObservationSpace()
{
  // 1.socket unique ID
  // 2.tcp env type: event-based = 0 / time-based = 1
  // 3.sim time in us
  // 4.node ID
  // 5.ssThresh
  // 6.cWnd
  // 7.segmentSize
  // 8.bytesInFlightSum
  // 9.bytesInFlightAvg
  // 10;segmentsAckedSum
  // 11segmentsAckedAvg
  // 12avgRtt
  // 13minRtt
  // 14avgInterTx
  // 15avgInterRx
  // 16throughput
  uint32_t parameterNum = 16;
  float low = 0.0;
  float high = 1000000000.0;
  std::vector<uint32_t> shape = {parameterNum,};
  std::string dtype = TypeNameGet<uint64_t> ();
  Ptr<OpenGymBoxSpace> box = CreateObject<OpenGymBoxSpace> (low, high, shape, dtype);
  return box;
}

/*
Collect observations
*/
Ptr<OpenGymDataContainer>
TcpTimeStepGymEnv::GetObservation()
{
  uint32_t parameterNum = 16;
  std::vector<uint32_t> shape = {parameterNum,};

  Ptr<OpenGymBoxContainer<uint64_t> > box = CreateObject<OpenGymBoxContainer<uint64_t>>
                                                                        (shape);
  
  box->AddValue(m_socketUuid);
  box->AddValue(1);
  box->AddValue(Simulator::Now().GetMicroSeconds ());
  box->AddValue(m_nodeId);
  
  //0
  box->AddValue(m_tcb->m_ssThresh);
  //1
  box->AddValue(m_tcb->m_cWnd);
  //2
  box->AddValue(m_tcb->m_segmentSize);

  //3.bytesInFlightSum
  uint64_t bytesInFlightSum = std::accumulate(m_bytesInFlight.begin(), m_bytesInFlight.end(), 
                                              0);
  box->AddValue(bytesInFlightSum);

  
  //4.bytesInFlightAvg
  uint64_t bytesInFlightAvg = 0;
  if (m_bytesInFlight.size()) {
    bytesInFlightAvg = bytesInFlightSum / m_bytesInFlight.size();
  }
  box->AddValue(bytesInFlightAvg);

  //5.segmentsAckedSum
   uint64_t segmentsAckedSum = std::accumulate(m_segmentsAcked.begin(),m_segmentsAcked.end(), 
                                               0);
  box->AddValue(segmentsAckedSum);

  //6.segmentsAckedAvg
  uint64_t segmentsAckedAvg = 0;
  if (m_segmentsAcked.size()) {
    segmentsAckedAvg = segmentsAckedSum / m_segmentsAcked.size();
  }
  box->AddValue(segmentsAckedAvg);

  //7.avgRtt
  // Time avgRtt = Seconds(0.0);
  // if(m_rttSampleNum) {
  //   avgRtt = m_rttSum / m_rttSampleNum;
  // }
  // box->AddValue(avgRtt.GetMicroSeconds ());
  box->AddValue(m_tcb->m_lastRtt.Get().GetMicroSeconds());

  //8.m_minRtt
   box->AddValue(m_tcb->m_minRtt.GetMicroSeconds ());

  // 9.avgInterTx
  Time avgInterTx = Seconds(0.0);
  if (m_interTxTimeNum) {
    avgInterTx = m_interTxTimeSum / m_interTxTimeNum;
  }
  box->AddValue(avgInterTx.GetMicroSeconds ());

  // 10.avgInterRx
  Time avgInterRx = Seconds(0.0);
  if (m_interRxTimeNum) {
    avgInterRx = m_interRxTimeSum / m_interRxTimeNum;
  }
  box->AddValue(avgInterRx.GetMicroSeconds ());

  //11.throughput  bytes/s
  uint64_t throughput = (segmentsAckedSum * m_tcb->m_segmentSize) / m_timeStep.GetSeconds();
  m_throughput=throughput;
  box->AddValue(throughput);

  if(throughput==0)box=old_box;
  else old_box=box;



  //EWMA avgInterTx
  // Time EWMA_InterTx=Seconds(0.0);
  // if(flag){
  //    EWMA_InterTx=m_last_interTxTime;
  //    m_EWMA_interTxTime=EWMA_InterTx;
  //    flag=0;
  // }
  // else{
  //    EWMA_InterTx=alpha*m_last_interTxTime + (1-alpha) * m_EWMA_interTxTime;
  //    m_EWMA_interTxTime = EWMA_InterTx;
  // }
  // //cout<<
  // box->AddValue(EWMA_InterTx.GetMicroSeconds ());

   //EWMA avgInterRx
  // Time EWMA_InterRx=Seconds(0.0);
  // if(flag1){
  //    EWMA_InterRx=m_last_interRxTime;
  //    m_EWMA_interRxTime=EWMA_InterRx;
  //    flag1=0;
  // }
  // else{
  //    EWMA_InterRx=alpha*m_last_interRxTime + (1-alpha) * m_EWMA_interRxTime;
  //    m_EWMA_interRxTime = EWMA_InterRx;
  // }
  // box->AddValue(EWMA_InterRx.GetMicroSeconds ());


/*---------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------*/

  // Update reward based on overall average of avgRtt over all steps so far
  // only when agent increases cWnd
  // TODO: this is not the right way of doing this.
  // place this somewhere else. see TcpEventGymEnv, how they've done it.

  // if (m_new_cWnd > m_old_cWnd && m_totalAvgRttSum > 0 && avgRtt > 0)  {
  //   // when agent increases cWnd
  //   if ((m_totalAvgRttSum / m_totalAvgRttNum) >= avgRtt)  {
  //     // give reward for decreasing avgRtt
  //     m_envReward = m_reward;
  //   } else {
  //     // give penalty for increasing avgRtt
  //     m_envReward = m_penalty;
  //   }
  // } else  {
  //   // agent has not increased cWnd
  //   m_envReward = 0;
  // }

  // // Update m_totalAvgRtSum and m_totalAvgRttNum
  // m_totalAvgRttSum += avgRtt;
  // m_totalAvgRttNum++;

  // m_old_cWnd = m_new_cWnd;
/*---------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------*/


  if(m_tcb!=nullptr){
  
  //uint64_t diff=m_current_rtt.GetMicroSeconds();
  uint64_t diff=(m_current_rtt-m_minRtt).GetMicroSeconds();
  uint64_t segmentsAckedSum = std::accumulate(m_segmentsAcked.begin(), m_segmentsAcked.end(), 0);
  
  float m_throughput = (segmentsAckedSum * m_tcb->m_segmentSize) / m_timeStep.GetSeconds();
  // std::cout<<"Throughput: "<<m_throughput<<"\n";
  // std::cout<<"Min RTT: "<<m_minRtt<<"\n";
  // std::cout<<"Current RTT: "<<m_current_rtt<<"\n";
 
  
  float utility_value=log((m_throughput*8)/(10*1e6))-0.1*log(diff);
  if(std::isinf(utility_value) or std::isnan(utility_value)){
   // utility_value=log((m_throughput*8)/(10*1e6));
   utility_value=old_U;
  }
    float delta=utility_value-old_U;
    if(delta>=0)m_reward=10;
    if(delta>=0 and delta<1)m_reward=5;
    if(delta>=-1 and delta<0)m_reward=-5;

    if(delta<-1)m_reward=-10;
    // std::cout<<"Utility value: "<<delta<<"\n";
    // std::cout<<"-----------------\n";

  
    old_U=utility_value;
  }
  // std::cout<<"Node ID: "<<m_nodeId<<"\n";
  // Print data
  NS_LOG_INFO ("MyGetObservation: " << box);

  m_bytesInFlight.clear();
  m_segmentsAcked.clear();
  //_utilities.clear();
  m_rttSampleNum = 0;
  m_rttSum = MicroSeconds (0.0);
  m_current_rtt=MicroSeconds(0.0);
  m_minRtt=MicroSeconds(0.0);
  m_throughput=0.0;
  //m_minRtt=0;
  m_interTxTimeNum = 0;
  m_interTxTimeSum = MicroSeconds (0.0);

  m_interRxTimeNum = 0;
  m_interRxTimeSum = MicroSeconds (0.0);
  

  m_last_interRxTime=MicroSeconds(0.0);
  m_last_interTxTime=MicroSeconds(0.0);
  m_EWMA_interRxTime=MicroSeconds(0.0);
  m_EWMA_interTxTime=MicroSeconds(0.0);

  flag=1;
  flag1=1;
  //old_box=box;
  return box;
}

void
TcpTimeStepGymEnv::TxPktTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>)
{
  // NS_LOG_FUNCTION (this);
  if ( m_lastPktTxTime > MicroSeconds(0.0) ) {
    Time interTxTime = Simulator::Now() - m_lastPktTxTime;
    m_last_interTxTime=interTxTime;
    m_interTxTimeSum += interTxTime;
    m_interTxTimeNum++;
  }

  m_lastPktTxTime = Simulator::Now();
}

void
TcpTimeStepGymEnv::RxPktTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>)
{
  // NS_LOG_FUNCTION (this);
  if ( m_lastPktRxTime > MicroSeconds(0.0) ) {
    Time interRxTime = Simulator::Now() - m_lastPktRxTime;
    m_last_interRxTime=interRxTime;
    m_interRxTimeSum +=  interRxTime;
    m_interRxTimeNum++;
  }

  m_lastPktRxTime = Simulator::Now();
}

uint32_t
TcpTimeStepGymEnv::GetSsThresh (Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight)
{
  // NS_LOG_FUNCTION (this);
  // NS_LOG_INFO(Simulator::Now() << " Node: " << m_nodeId << " GetSsThresh, BytesInFlight: " << bytesInFlight);
  m_tcb = tcb;
  m_bytesInFlight.push_back(bytesInFlight);

  if (!m_started) {
    m_started = true;
    Notify();
    ScheduleNextStateRead();
  }

  // action
  return m_new_ssThresh;
  
}

void
TcpTimeStepGymEnv::IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
  // NS_LOG_FUNCTION (this);
  // NS_LOG_INFO(Simulator::Now() << " Node: " << m_nodeId << " IncreaseWindow, SegmentsAcked: " << segmentsAcked);
  m_tcb = tcb;
  m_segmentsAcked.push_back(segmentsAcked);
  m_bytesInFlight.push_back(tcb->m_bytesInFlight);

  if (!m_started) {
    m_started = true;
    Notify();
    ScheduleNextStateRead();
  }
  // action
  tcb->m_cWnd = m_new_cWnd;
  NS_LOG_INFO("New congestion window: "<<m_new_cWnd);
  std::cout<<"\n\n";
  
}

void
TcpTimeStepGymEnv::PktsAcked (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt)
{
  // NS_LOG_FUNCTION (this);
  // NS_LOG_INFO(Simulator::Now() << " Node: " << m_nodeId << " PktsAcked, SegmentsAcked: " << segmentsAcked << " Rtt: " << rtt);
  m_tcb = tcb;
  m_rttSum += rtt;

  m_minRtt=std::min(m_minRtt,rtt);
  m_current_rtt=rtt;
  m_rttSampleNum++;

 // tcb->m_cWnd=tcb->m_cWnd+(m_new_cWnd/tcb->m_cWnd); //set new congestion window
}

void
TcpTimeStepGymEnv::CongestionStateSet (Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCongState_t newState)
{
  // NS_LOG_FUNCTION (this);
  std::string stateName = GetTcpCongStateName(newState);
  // std::cout<<cnt<<"\n";
  // std::cout<<"Congestion occurred\n";
  // std::cout<<"Node ID: "<<m_nodeId<<"\n";
  // std::cout<<"Old CWND: "<<tcb->m_cWnd<<"\n";
  // std::cout<<"New CWND: "<<m_new_cWnd<<"\n";
  
  //tcb->m_cWnd = m_new_cWnd;
  // NS_LOG_INFO(Simulator::Now() << " Node: " << m_nodeId << " CongestionStateSet: " << newState << " " << stateName);
  m_tcb = tcb;
 // cnt=cnt+1;
}

void
TcpTimeStepGymEnv::CwndEvent (Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event)
{
  // NS_LOG_FUNCTION (this);
  std::string eventName = GetTcpCAEventName(event);
  // NS_LOG_INFO(Simulator::Now() << " Node: " << m_nodeId << " CwndEvent: " << event << " " << eventName);
}

} // namespace ns3
