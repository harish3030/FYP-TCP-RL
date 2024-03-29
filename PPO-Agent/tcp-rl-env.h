#ifndef TCP_RL_ENV_H
#define TCP_RL_ENV_H
#include "ns3/opengym-module.h"
#include "ns3/tcp-socket-base.h"
#include <vector>
namespace ns3 {

class Packet;
class TcpHeader;
class TcpSocketBase;
class Time;


class TcpGymEnv : public OpenGymEnv
{
public:
  TcpGymEnv ();
  virtual ~TcpGymEnv ();
  static TypeId GetTypeId (void);
  virtual void DoDispose ();

  void SetNodeId(uint32_t id);
  void SetSocketUuid(uint32_t id);

  std::string GetTcpCongStateName(const TcpSocketState::TcpCongState_t state);
  std::string GetTcpCAEventName(const TcpSocketState::TcpCAEvent_t event);

  // OpenGym interface
  virtual Ptr<OpenGymSpace> GetActionSpace();
  virtual bool GetGameOver();
  virtual float GetReward()=0;
  virtual std::string GetExtraInfo();
  virtual bool ExecuteActions(Ptr<OpenGymDataContainer> action);

  virtual Ptr<OpenGymSpace> GetObservationSpace() = 0;
  virtual Ptr<OpenGymDataContainer> GetObservation() = 0;

  // trace packets, e.g. for calculating inter tx/rx time
  virtual void TxPktTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>) = 0;
  virtual void RxPktTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>) = 0;

  // TCP congestion control interface
  virtual uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight) = 0;
  virtual void IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked) = 0;
  // optional functions used to collect obs
  virtual void PktsAcked (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt) = 0;
  virtual void CongestionStateSet (Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCongState_t newState) = 0;
  virtual void CwndEvent (Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event) = 0;

  typedef enum
  {
    GET_SS_THRESH = 0,
    INCREASE_WINDOW,
    PKTS_ACKED,
    CONGESTION_STATE_SET,
    CWND_EVENT,
  } CalledFunc_t;

protected:
  uint32_t m_nodeId;
  uint32_t m_socketUuid;

  // state
  // obs has to be implemented in child class

  // game over
  bool m_isGameOver;

  // reward
  float m_envReward;

  // extra info
  std::string m_info;

  // actions
  uint32_t m_new_ssThresh;
  uint32_t m_new_cWnd;
};


class TcpEventGymEnv : public TcpGymEnv
{
public:
  TcpEventGymEnv ();
  virtual ~TcpEventGymEnv ();
  static TypeId GetTypeId (void);
  virtual void DoDispose ();

  void SetReward(float value);
  void SetPenalty(float value);

  // OpenGym interface
  virtual Ptr<OpenGymSpace> GetObservationSpace();
  float GetReward();
  Ptr<OpenGymDataContainer> GetObservation();

  // trace packets, e.g. for calculating inter tx/rx time
  virtual void TxPktTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>);
  virtual void RxPktTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>);

  // TCP congestion control interface
  virtual uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight);
  virtual void IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked);
  // optional functions used to collect obs
  virtual void PktsAcked (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt);
  virtual void CongestionStateSet (Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCongState_t newState);
  virtual void CwndEvent (Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event);

private:
  // state
  CalledFunc_t m_calledFunc;
  Ptr<const TcpSocketState> m_tcb;
  uint32_t m_bytesInFlight;
  uint32_t m_segmentsAcked;
  Time m_rtt;
  TcpSocketState::TcpCongState_t m_newState;
  TcpSocketState::TcpCAEvent_t m_event;

  // reward
  float m_reward;
  float m_penalty;
};


class TcpTimeStepGymEnv : public TcpGymEnv
{
public:
  TcpTimeStepGymEnv ();

  virtual ~TcpTimeStepGymEnv ();
  static TypeId GetTypeId (void);
  virtual void DoDispose ();

  void SetDuration(Time value);
  void SetTimeStep(Time value);
  void SetReward(float value);
  void SetPenalty(float value);

  // OpenGym interface
  virtual Ptr<OpenGymSpace> GetObservationSpace();
  float GetReward();
  Ptr<OpenGymDataContainer> GetObservation();

  // trace packets, e.g. for calculating inter tx/rx time
  virtual void TxPktTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>);
  virtual void RxPktTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>);

  // TCP congestion control interface
  virtual uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight);
  virtual void IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked);
  // optional functions used to collect obs
  virtual void PktsAcked (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt);
  virtual void CongestionStateSet (Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCongState_t newState);
  virtual void CwndEvent (Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event);


  void ScheduleNextStateRead();
private:
  bool m_started {false};
  Time m_duration;
  Time m_timeStep;

  float old_U=0.0;
  float alpha=0.1;
  bool flag=1;
  bool flag1=1;
  int cnt=0;
  // state
  Ptr<const TcpSocketState> m_tcb;
  Ptr<OpenGymBoxContainer<uint64_t> > old_box;
  std::vector<uint32_t> m_bytesInFlight;
  std::vector<uint32_t> m_segmentsAcked;
  
  Time m_last_interTxTime {MicroSeconds(0.0)};
  Time m_last_interRxTime {MicroSeconds (0.0)};
  Time m_EWMA_interTxTime {MicroSeconds(0.0)};
  Time m_EWMA_interRxTime {MicroSeconds(0.0)};
  uint64_t m_rttSampleNum {0};
  Time m_rttSum {MicroSeconds (0.0)};
  Time m_current_rtt {MicroSeconds (0.0)};
  
  float m_throughput {0.0};
  Time m_lastPktTxTime {MicroSeconds(0.0)};
  Time m_lastPktRxTime {MicroSeconds(0.0)};
  uint64_t m_interTxTimeNum {0};
  Time m_interTxTimeSum {MicroSeconds (0.0)};
 
  uint64_t m_interRxTimeNum {0};
  Time m_interRxTimeSum {MicroSeconds (0.0)};
 
  Time m_prevAvgRtt {MicroSeconds (0.0)};
  Time m_totalAvgRttSum {MicroSeconds (0.0)};
  uint64_t m_totalAvgRttNum {0};
  uint32_t m_old_cWnd {0};
  Time m_minRtt {Time::Max()}; 

  // reward
  float m_reward;
  float m_penalty;
};



} // namespace ns3

#endif /* TCP_RL_ENV_H */
