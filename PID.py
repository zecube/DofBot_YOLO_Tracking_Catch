#!/usr/bin/env python
# coding: utf-8
#  PID control
# *****************************************************************#
#                   Incremental PID system                           #
# *****************************************************************#
class IncrementalPID:
    def __init__(self, P, I, D):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.PIDOutput = 0.0  
        self.SystemOutput = 0.0  
        self.LastSystemOutput = 0.0  
        self.Error = 0.0 
        self.LastError = 0.0
        self.LastLastError = 0.0

    # Set PID controller parameters
    def SetStepSignal(self, StepSignal):
        self.Error = StepSignal - self.SystemOutput
        IncrementValue = self.Kp * (self.Error - self.LastError) + \
                         self.Ki * self.Error + \
                         self.Kd * (self.Error - 2 * self.LastError + self.LastLastError)
        self.PIDOutput += IncrementValue
        self.LastLastError = self.LastError
        self.LastError = self.Error
    # Set InertiaTime as inertial time constant in the first-order inertial link system
    def SetInertiaTime(self, InertiaTime, SampleTime):
        self.SystemOutput = (InertiaTime * self.LastSystemOutput + \
                             SampleTime * self.PIDOutput) / (SampleTime + InertiaTime)
        self.LastSystemOutput = self.SystemOutput
# *****************************************************************#
#                      Position type PID system                   #
# *****************************************************************#
class PositionalPID:
    def __init__(self, P, I, D):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.SystemOutput = 0.0
        self.ResultValueBack = 0.0
        self.PidOutput = 0.0
        self.PIDErrADD = 0.0
        self.ErrBack = 0.0
    # Set PID controller parameters
    def SetStepSignal(self, StepSignal):
        Err = StepSignal - self.SystemOutput
        KpWork = self.Kp * Err
        KiWork = self.Ki * self.PIDErrADD
        KdWork = self.Kd * (Err - self.ErrBack)
        self.PidOutput = KpWork + KiWork + KdWork
        self.PIDErrADD += Err
        self.ErrBack = Err
    # Set InertiaTime as inertial time constant in the first-order inertial link system
    def SetInertiaTime(self, InertiaTime, SampleTime):
        self.SystemOutput = (InertiaTime * self.ResultValueBack + \
                             SampleTime * self.PidOutput) / (SampleTime + InertiaTime)
        self.ResultValueBack = self.SystemOutput
