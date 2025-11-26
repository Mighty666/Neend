'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import { ArrowDownTrayIcon, ShareIcon, ChevronLeftIcon } from '@heroicons/react/24/outline'
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'

// Mock data
const sessionData = {
  id: '1',
  date: 'November 22, 2025',
  duration: 7.5,
  ahi: 12.4,
  severity: 'Mild OSA',
  apnea: 23,
  hypopnea: 47,
  snoring: 89,
  normal: 801,
}

const hourlyData = Array.from({ length: 8 }, (_, i) => ({
  hour: `${i + 11 > 12 ? i - 1 : i + 11}${i + 11 >= 12 ? 'AM' : 'PM'}`,
  ahi: Math.random() * 20 + (i > 2 && i < 6 ? 10 : 0),
}))

const pieData = [
  { name: 'Normal', value: sessionData.normal, color: '#4ade80' },
  { name: 'Snoring', value: sessionData.snoring, color: '#facc15' },
  { name: 'Hypopnea', value: sessionData.hypopnea, color: '#fb923c' },
  { name: 'Apnea', value: sessionData.apnea, color: '#f87171' },
]

export default function ReportPage({ params }: { params: { id: string } }) {
  return (
    <div className="min-h-screen bg-dark-950">
      <header className="glass-dark sticky top-0 z-50">
        <div className="max-w-4xl mx-auto px-6 py-4 flex justify-between items-center">
          <Link href="/history" className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors">
            <ChevronLeftIcon className="w-5 h-5" />
            Back
          </Link>
          <div className="flex gap-2">
            <button className="p-2 glass rounded-lg hover:bg-white/10 transition-colors">
              <ShareIcon className="w-5 h-5" />
            </button>
            <button className="p-2 glass rounded-lg hover:bg-white/10 transition-colors">
              <ArrowDownTrayIcon className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-2xl font-bold mb-1">Sleep Report</h1>
          <p className="text-gray-400">{sessionData.date}</p>
        </motion.div>

        {/* Summary Cards */}
        <div className="grid md:grid-cols-4 gap-4 mb-8">
          {[
            { label: 'AHI Score', value: sessionData.ahi.toFixed(1), color: 'text-blue-400' },
            { label: 'Severity', value: sessionData.severity, color: 'text-yellow-400' },
            { label: 'Duration', value: `${sessionData.duration}h`, color: 'text-purple-400' },
            { label: 'Total Events', value: sessionData.apnea + sessionData.hypopnea, color: 'text-red-400' },
          ].map((stat, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
              className="glass rounded-xl p-4"
            >
              <p className="text-sm text-gray-400 mb-1">{stat.label}</p>
              <p className={`text-2xl font-bold ${stat.color}`}>{stat.value}</p>
            </motion.div>
          ))}
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
            <h3 className="text-lg font-semibold mb-4">Hourly AHI</h3>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={hourlyData}>
                  <defs>
                    <linearGradient id="ahiGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="hour" stroke="#6b7280" fontSize={10} tickLine={false} />
                  <YAxis stroke="#6b7280" fontSize={10} tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1a1a1f', border: '1px solid #2f416d', borderRadius: '8px' }}
                  />
                  <Area type="monotone" dataKey="ahi" stroke="#3b82f6" fill="url(#ahiGrad)" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </motion.div>

          {/* Event Distribution */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="glass rounded-xl p-6"
          >
            <h3 className="text-lg font-semibold mb-4">Event Distribution</h3>
            <div className="h-48 flex items-center justify-center">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={60}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={index} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1a1a1f', border: '1px solid #2f416d', borderRadius: '8px' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex justify-center gap-4 text-xs">
              {pieData.map((item) => (
                <div key={item.name} className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }} />
                  {item.name}
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* Recommendations */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="glass rounded-xl p-6"
        >
          <h3 className="text-lg font-semibold mb-4">Recommendations</h3>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <div>
                <p className="font-medium">Mild OSA Detected</p>
                <p className="text-sm text-gray-400">Consider consulting a sleep specialist for evaluation.</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div>
                <p className="font-medium">Sleep Position</p>
                <p className="text-sm text-gray-400">Try sleeping on your side to reduce apnea events.</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div>
                <p className="font-medium">Lifestyle Changes</p>
                <p className="text-sm text-gray-400">Weight management and avoiding alcohol before bed may help.</p>
              </div>
            </div>
          </div>
        </motion.div>
      </main>
    </div>
  )
}
