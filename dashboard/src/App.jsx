import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, 
  TrendingUp, 
  TrendingDown, 
  AlertCircle, 
  Play, 
  Square, 
  RefreshCw,
  History,
  Zap,
  ShieldCheck
} from 'lucide-react';
import './App.css';

const WS_URL = 'ws://localhost:8000/ws/signal-stream';

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [candles, setCandles] = useState([]);
  const [signals, setSignals] = useState([]);
  const [trades, setTrades] = useState([]);
  const [latestPrice, setLatestPrice] = useState(0);
  const [paperTradeStatus, setPaperTradeStatus] = useState('IDLE'); // IDLE, RUNNING, STOPPING
  const [jobs, setJobs] = useState([]);
  
  const ws = useRef(null);

  useEffect(() => {
    connectWS();
    fetchJobs();
    const interval = setInterval(fetchJobs, 5000);
    return () => {
      clearInterval(interval);
      if (ws.current) ws.current.close();
    };
  }, []);

  const connectWS = () => {
    console.log('Connecting to WS...');
    ws.current = new WebSocket(WS_URL);

    ws.current.onopen = () => {
      setIsConnected(true);
      console.log('Connected to WebSocket');
    };

    ws.current.onclose = () => {
      setIsConnected(false);
      console.log('Disconnected from WebSocket');
      // Reconnect after 3 seconds
      setTimeout(connectWS, 3000);
    };

    ws.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      handleMessage(message);
    };
  };

  const handleMessage = (message) => {
    const { type, data } = message;
    
    if (type === 'candle') {
      setLatestPrice(data.close);
      setCandles(prev => [data, ...prev].slice(0, 20));
    } else if (type === 'signal') {
      setSignals(prev => [data, ...prev].slice(0, 10));
    } else if (type === 'trade') {
      setTrades(prev => [data, ...prev].slice(0, 20));
    }
  };

  const fetchJobs = async () => {
    try {
      const res = await fetch('/api/jobs');
      const data = await res.json();
      setJobs(data.jobs);
      
      const isRunning = data.jobs.some(job => job.key.includes('paper-trade'));
      setPaperTradeStatus(isRunning ? 'RUNNING' : 'IDLE');
    } catch (err) {
      console.error('Failed to fetch jobs', err);
    }
  };

  const startPaperTrade = async () => {
    try {
      setPaperTradeStatus('STARTING');
      await fetch('/api/paper-trade/start', { method: 'POST' });
      fetchJobs();
    } catch (err) {
      alert('Failed to start paper trade');
      setPaperTradeStatus('IDLE');
    }
  };

  const stopPaperTrade = async () => {
    try {
      const ptJob = jobs.find(job => job.key.includes('paper-trade'));
      if (!ptJob) return;
      
      setPaperTradeStatus('STOPPING');
      await fetch('/api/jobs/stop', { 
        method: 'POST', 
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pid: ptJob.pid })
      });
      setTimeout(fetchJobs, 2000);
    } catch (err) {
      alert('Failed to stop paper trade');
    }
  };

  return (
    <div className="dashboard">
      <header className="header">
        <div className="logo">
          <Zap className="logo-icon" size={24} />
          <h1>Autopilot AI</h1>
        </div>
        <div className="system-status">
          <div className={`status-badge ${isConnected ? 'online' : 'offline'}`}>
            <Activity size={16} />
            <span>{isConnected ? 'LIVE' : 'DISCONNECTED'}</span>
          </div>
          <div className="price-ticker">
            <span className="symbol">ETH/USDT</span>
            <span className="price">${latestPrice ? latestPrice.toLocaleString() : '---'}</span>
          </div>
        </div>
      </header>

      <main className="main-grid">
        {/* Left Column: Market & Controls */}
        <div className="col">
          <section className="card controls-card">
            <div className="card-header">
              <ShieldCheck size={20} />
              <h2>Trading Controls</h2>
            </div>
            <div className="card-content">
              <div className="status-display">
                <span className="label">Paper Trade Status</span>
                <span className={`value status-${paperTradeStatus.toLowerCase()}`}>
                  {paperTradeStatus}
                </span>
              </div>
              <div className="actions">
                {paperTradeStatus === 'IDLE' ? (
                  <button className="btn btn-primary" onClick={startPaperTrade}>
                    <Play size={18} /> Start Paper Trading
                  </button>
                ) : (
                  <button className="btn btn-danger" onClick={stopPaperTrade} disabled={paperTradeStatus === 'STOPPING'}>
                    <Square size={18} /> Stop Paper Trading
                  </button>
                )}
              </div>
            </div>
          </section>

          <section className="card signal-card">
            <div className="card-header">
              <AlertCircle size={20} />
              <h2>Latest AI Signals</h2>
            </div>
            <div className="card-content list-container">
              {signals.length === 0 ? (
                <div className="empty-state">Waiting for signals...</div>
              ) : (
                signals.map((sig, i) => (
                  <div key={i} className={`list-item signal-${sig.action.toLowerCase()}`}>
                    <div className="sig-main">
                      <span className="sig-action">{sig.action}</span>
                      <span className="sig-conf">{(sig.confidence * 100).toFixed(1)}% confidence</span>
                    </div>
                    <div className="sig-meta">
                      <span>Price: ${sig.price || sig.features?.[3]}</span>
                      <span>{new Date(sig.timestamp).toLocaleTimeString()}</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>
        </div>

        {/* Right Column: Trade History & Activity */}
        <div className="col">
          <section className="card history-card">
            <div className="card-header">
              <History size={20} />
              <h2>Recent Activity</h2>
            </div>
            <div className="card-content list-container">
              {trades.length === 0 ? (
                <div className="empty-state">No trades executed in this session</div>
              ) : (
                trades.map((trade, i) => (
                  <div key={i} className={`list-item trade-${trade.side.toLowerCase()}`}>
                    <div className="trade-main">
                      <span className="trade-side">{trade.side} {trade.action}</span>
                      <span className="trade-price">${trade.price.toLocaleString()}</span>
                    </div>
                    <div className="trade-meta">
                      <span className={trade.pnl_pct >= 0 ? 'pos' : 'neg'}>
                        {trade.pnl_pct ? `${(trade.pnl_pct * 100).toFixed(2)}%` : '---'}
                      </span>
                      <span>{new Date(trade.timestamp).toLocaleTimeString()}</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>

          <section className="card market-card">
            <div className="card-header">
              <RefreshCw size={20} />
              <h2>Market Feed</h2>
            </div>
            <div className="card-content list-container">
              {candles.map((c, i) => (
                <div key={i} className="list-item candle-item">
                  <span>{new Date(c.time).toLocaleTimeString()}</span>
                  <span className="c-price">${c.close.toLocaleString()}</span>
                  <span className={`c-change ${c.close >= c.open ? 'pos' : 'neg'}`}>
                    {((c.close/c.open - 1) * 100).toFixed(2)}%
                  </span>
                </div>
              ))}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}

export default App;
