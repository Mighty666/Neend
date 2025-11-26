import { useState, useRef, useCallback } from 'react'

interface AudioRecorderState {
  isRecording: boolean
  isPaused: boolean
  duration: number
  error: string | null
}

export function useAudioRecorder() {
  const [state, setState] = useState<AudioRecorderState>({
    isRecording: false,
    isPaused: false,
    duration: 0,
    error: null,
  })

  const mediaRecorder = useRef<MediaRecorder | null>(null)
  const audioChunks = useRef<Blob[]>([])
  const timerRef = useRef<NodeJS.Timeout | null>(null)
  const onChunkCallback = useRef<((chunk: Blob) => void) | null>(null)

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000,
        },
      })

      const recorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      })

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.current.push(event.data)
          if (onChunkCallback.current) {
            onChunkCallback.current(event.data)
          }
        }
      }

      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop())
      }

      recorder.start(1000) // Collect data every second
      mediaRecorder.current = recorder

      // Start timer
      timerRef.current = setInterval(() => {
        setState((prev) => ({ ...prev, duration: prev.duration + 1 }))
      }, 1000)

      setState({
        isRecording: true,
        isPaused: false,
        duration: 0,
        error: null,
      })
    } catch (err) {
      setState((prev) => ({
        ...prev,
        error: 'Failed to access microphone',
      }))
    }
  }, [])

  const stopRecording = useCallback((): Blob | null => {
    if (mediaRecorder.current && state.isRecording) {
      mediaRecorder.current.stop()

      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }

      setState({
        isRecording: false,
        isPaused: false,
        duration: 0,
        error: null,
      })

      const blob = new Blob(audioChunks.current, { type: 'audio/webm' })
      audioChunks.current = []
      return blob
    }
    return null
  }, [state.isRecording])

  const pauseRecording = useCallback(() => {
    if (mediaRecorder.current && state.isRecording) {
      mediaRecorder.current.pause()
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
      setState((prev) => ({ ...prev, isPaused: true }))
    }
  }, [state.isRecording])

  const resumeRecording = useCallback(() => {
    if (mediaRecorder.current && state.isPaused) {
      mediaRecorder.current.resume()
      timerRef.current = setInterval(() => {
        setState((prev) => ({ ...prev, duration: prev.duration + 1 }))
      }, 1000)
      setState((prev) => ({ ...prev, isPaused: false }))
    }
  }, [state.isPaused])

  const onChunk = useCallback((callback: (chunk: Blob) => void) => {
    onChunkCallback.current = callback
  }, [])

  return {
    ...state,
    startRecording,
    stopRecording,
    pauseRecording,
    resumeRecording,
    onChunk,
  }
}
