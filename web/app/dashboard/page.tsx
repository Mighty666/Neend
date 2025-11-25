'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'
import { MoonIcon, SunIcon, BellAlertIcon, ChartBarIcon, ClockIcon, ArrowTrendingUpIcon, ArrowTrendingDownIcon } from '@heroicons/react/24/outline'

// Mock data for demo
const generateMockData = () => {
  const data = []
  for (let i = 0; i < 24; i++) {
    data.push({
      hour: `${i}:00`,
      ahi: Math.random() * 15 + (i > 1 && i < 6 ? 10 : 0),
      events: Math.floor(Math.random() * 5),
    })
  }
  return data
}

const weeklyData = [
  { day: 'Mon', ahi: 12.3 },
  { day: 'Tue', ahi: 14.1 },
  { day: 'Wed', ahi: 11.8 },
  { day: 'Thu', ahi: 13.5 },
  { day: 'Fri', ahi: 10.2 },
  { day: 'Sat', ahi: 15.7 },
  { day: 'Sun', ahi: 12.9 },
]

export default function Dashboard() {
  const [currentAHI, setCurrentAHI] = useState(12.4)
  const [isRecording, setIsRecording] = useState(false)
  const [hourlyData, setHourlyData] = useState(generateMockData())

  const severity = currentAHI < 5 ? 'Normal' : currentAHI < 15 ? 'Mild' : currentAHI < 30 ? 'Moderate' : 'Severe'
  const severityColor = currentAHI < 5 ? 'text-green-400' : currentAHI < 15 ? 'text-yellow-400' : currentAHI < 30 ? 'text-orange-400' : 'text-red-400'

  return (
    <div className="min-h-screen bg-dark-950">
      {/* Header */}
      <header className="glass-dark sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="text-xl font-bold">
            <span className="text-gradient">Neend</span>
            <span className="text-white">AI</span>
          </div>
          <div className="flex items-center gap-4">
            <button className="p-2 hover:bg-white/5 rounded-lg transition-colors">
              <BellAlertIcon className="w-5 h-5 text-gray-400" />
            </button>
            <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-sm font-medium">
              M
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Welcome */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-2xl font-bold mb-1">Good Evening, Maanas</h1>
          <p className="text-gray-400">Here's your sleep analysis summary</p>
        </motion.div>

        {/* Stats Grid */}
        <div className="grid md:grid-cols-4 gap-4 mb-8">
          {/* AHI Score */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="glass rounded-xl p-6 md:col-span-2"
          >
            <div className="flex justify-between items-start mb-4">
              <div>
                <p className="text-sm text-gray-400 mb-1">Current AHI Score</p>
                <p className="text-4xl font-bold">{currentAHI.toFixed(1)}</p>
              </div>
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${severityColor} bg-white/5`}>
                {severity}
              </span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <ArrowTrendingDownIcon className="w-4 h-4 text-green-400" />
              <span className="text-green-400">-2.3</span>
              <span className="text-gray-500">from last week</span>
            </div>
          </motion.div>

          {/* Sleep Duration */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="glass rounded-xl p-6"
          >
            <div className="flex items-center gap-2 mb-3">
              <ClockIcon className="w-5 h-5 text-blue-400" />
              <span className="text-sm text-gray-400">Sleep Duration</span>
            </div>
            <p className="text-2xl font-bold">7h 23m</p>
            <p className="text-sm text-gray-500 mt-1">Last night</p>
          </motion.div>

          {/* Events */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="glass rounded-xl p-6"
          >
            <div className="flex items-center gap-2 mb-3">
              <ChartBarIcon className="w-5 h-5 text-purple-400" />
              <span className="text-sm text-gray-400">Total Events</span>
            </div>
            <p className="text-2xl font-bold">47</p>
            <p className="text-sm text-gray-500 mt-1">Apnea + Hypopnea</p>
          </motion.div>
        </div>

        {/* Charts */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {/* Hourly AHI */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="glass rounded-xl p-6"
          >
            <h3 className="text-lg font-semibold mb-4">Hourly AHI - Last Night</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={hourlyData}>
                  <defs>
                    <linearGradient id="ahiGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <XAxis
                    dataKey="hour"
                    stroke="#6b7280"
                    fontSize={12}
                    tickLine={false}
                  />
                  <YAxis
                    stroke="#6b7280"
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1a1a1f',
                      border: '1px solid #2f416d',
                      borderRadius: '8px'
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="ahi"
                    stroke="#3b82f6"
                    fill="url(#ahiGradient)"
                    strokeWidth={2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </motion.div>

          {/* Weekly Trend */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="glass rounded-xl p-6"
          >
            <h3 className="text-lg font-semibold mb-4">Weekly Trend</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={weeklyData}>
                  <XAxis
                    dataKey="day"
                    stroke="#6b7280"
                    fontSize={12}
                    tickLine={false}
                  />
                  <YAxis
                    stroke="#6b7280"
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1a1a1f',
                      border: '1px solid #2f416d',
                      borderRadius: '8px'
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="ahi"
                    stroke="#8b5cf6"
                    strokeWidth={2}
                    dot={{ fill: '#8b5cf6', strokeWidth: 2 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </motion.div>
        </div>

        {/* Start Recording */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="glass rounded-xl p-8 text-center"
        >
          <div className="mb-6">
            {isRecording ? (
              <motion.div
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
                className="w-20 h-20 mx-auto bg-red-500/20 rounded-full flex items-center justify-center"
              >
                <div className="w-12 h-12 bg-red-500 rounded-full animate-pulse" />
              </motion.div>
            ) : (
              <div className="w-20 h-20 mx-auto bg-blue-500/20 rounded-full flex items-center justify-center">
                <MoonIcon className="w-10 h-10 text-blue-400" />
              </div>
            )}
          </div>
          <h3 className="text-xl font-semibold mb-2">
            {isRecording ? 'Recording in Progress...' : 'Ready to Sleep?'}
          </h3>
          <p className="text-gray-400 mb-6">
            {isRecording
              ? 'Analyzing your breathing patterns in real-time'
              : 'Start recording to analyze your sleep patterns tonight'
            }
          </p>
          <button
            onClick={() => setIsRecording(!isRecording)}
            className={`px-8 py-3 rounded-xl font-semibold transition-colors ${
              isRecording
                ? 'bg-red-600 hover:bg-red-500'
                : 'bg-blue-600 hover:bg-blue-500'
            }`}
          >
            {isRecording ? 'Stop Recording' : 'Start Sleep Analysis'}
          </button>
        </motion.div>
      </main>
    </div>
  )
}
