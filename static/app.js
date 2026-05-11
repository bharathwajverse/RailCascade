/**
 * RailCascade Mini V2 -- Frontend Application
 * Canvas-based graph visualization with real-time train animation
 */

// =============================================================
// Constants
// =============================================================

const API_BASE = '';

const TRAIN_COLORS = [
    '#22d3ee', // cyan
    '#a78bfa', // purple
    '#fbbf24', // amber
    '#34d399', // emerald
    '#fb7185', // rose
    '#60a5fa', // blue
    '#f472b6', // pink
    '#818cf8', // indigo
];

const NODE_COLORS = {
    source: '#22d3ee',
    junction: '#a78bfa',
    corridor: '#fbbf24',
    terminal: '#34d399',
};

const NODE_RADIUS = 24;
const TRAIN_RADIUS = 8;

// =============================================================
// State
// =============================================================

let graph = null;         // { nodes: [], edges: [] }
let envState = null;      // Full state from API
let prevTrainPositions = {}; // For interpolation
let animProgress = 1.0;   // 0..1 interpolation
let animStartTime = 0;
const ANIM_DURATION = 300; // ms

let autoInterval = null;
let stepHistory = [];
let totalConflicts = 0;
let sessionId = null;   // Session ID from /api/reset

// =============================================================
// DOM References
// =============================================================

const canvas = document.getElementById('graph-canvas');
const ctx = canvas.getContext('2d');
const overlay = document.getElementById('canvas-overlay');

const btnReset = document.getElementById('btn-reset');
const btnStep = document.getElementById('btn-step');
const btnAuto = document.getElementById('btn-auto');
const btnPause = document.getElementById('btn-pause');
const taskSelect = document.getElementById('task-select');
const speedSlider = document.getElementById('speed-slider');
const speedLabel = document.getElementById('speed-label');

const scoreValue = document.getElementById('score-value');
const scoreFill = document.getElementById('score-fill');
const statTimestep = document.getElementById('stat-timestep');
const statMaxsteps = document.getElementById('stat-maxsteps');
const statDelay = document.getElementById('stat-delay');
const statConflicts = document.getElementById('stat-conflicts');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');

const trainTableBody = document.getElementById('train-table-body');
const eventLog = document.getElementById('event-log');
const timelineBars = document.getElementById('timeline-bars');

const statusDot = document.querySelector('.status-dot');
const statusText = document.querySelector('.status-text');

// =============================================================
// Canvas Setup
// =============================================================

function resizeCanvas() {
    const section = document.getElementById('canvas-section');
    const dpr = window.devicePixelRatio || 1;
    canvas.width = section.clientWidth * dpr;
    canvas.height = section.clientHeight * dpr;
    canvas.style.width = section.clientWidth + 'px';
    canvas.style.height = section.clientHeight + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

window.addEventListener('resize', () => {
    resizeCanvas();
    render();
});

// =============================================================
// API Calls
// =============================================================

async function apiGet(path) {
    const res = await fetch(API_BASE + path);
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json();
}

async function apiPost(path, body = {}) {
    const res = await fetch(API_BASE + path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `API error: ${res.status}`);
    }
    return res.json();
}

async function loadGraph() {
    graph = await apiGet('/api/graph');
}

async function resetEnv(task) {
    const data = await apiPost('/api/reset', { task });
    sessionId = data.session_id;
    envState = data.state;
    stepHistory = [];
    totalConflicts = 0;
    prevTrainPositions = {};
    animProgress = 1.0;

    // Save initial positions
    if (envState && envState.trains) {
        envState.trains.forEach(t => {
            prevTrainPositions[t.id] = t.position;
        });
    }

    return data;
}

async function stepEnv() {
    const data = await apiPost('/api/auto_step', { session_id: sessionId });
    
    // Save previous positions for animation
    if (envState && envState.trains) {
        envState.trains.forEach(t => {
            prevTrainPositions[t.id] = t.position;
        });
    }

    envState = data.state;
    animProgress = 0;
    animStartTime = performance.now();

    // Track history - use info.new_delay and info.conflicts (actual env field names)
    const inf = data.info || {};
    if (envState) {
        stepHistory.push({
            step: envState.timestep,
            delay: inf.new_delay || 0,
            conflicts: inf.conflicts || 0,
            totalDelay: inf.total_delay != null ? inf.total_delay : 0,
        });
    }
    totalConflicts += inf.conflicts || 0;

    return data;
}

// =============================================================
// Coordinate Mapping
// =============================================================

function getNodeScreenPos(nodeId) {
    if (!graph) return { x: 0, y: 0 };
    const node = graph.nodes.find(n => n.id === nodeId);
    if (!node) return { x: 0, y: 0 };

    const section = document.getElementById('canvas-section');
    const w = section.clientWidth;
    const h = section.clientHeight;

    // Graph coords are designed for ~960x520 base
    const scaleX = w / 960;
    const scaleY = h / 520;
    const scale = Math.min(scaleX, scaleY) * 0.9;
    const offsetX = (w - 960 * scale) / 2;
    const offsetY = (h - 520 * scale) / 2;

    return {
        x: node.x * scale + offsetX,
        y: node.y * scale + offsetY,
    };
}

function getTrainScreenPos(train) {
    const currPos = getNodeScreenPos(train.position);
    
    if (animProgress >= 1.0 || !prevTrainPositions[train.id]) {
        return currPos;
    }

    const prevNodeId = prevTrainPositions[train.id];
    if (prevNodeId === train.position) return currPos;

    const prevPos = getNodeScreenPos(prevNodeId);
    const t = easeOutCubic(animProgress);
    return {
        x: prevPos.x + (currPos.x - prevPos.x) * t,
        y: prevPos.y + (currPos.y - prevPos.y) * t,
    };
}

function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
}

// =============================================================
// Rendering
// =============================================================

function render() {
    if (!canvas.width) return;

    const section = document.getElementById('canvas-section');
    const w = section.clientWidth;
    const h = section.clientHeight;

    ctx.clearRect(0, 0, w, h);

    // Background grid
    drawGrid(w, h);

    if (!graph) return;

    // Draw edges
    drawEdges();

    // Draw nodes
    drawNodes();

    // Draw trains
    if (envState && envState.trains) {
        drawTrains();
    }
}

function drawGrid(w, h) {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.02)';
    ctx.lineWidth = 1;
    const spacing = 40;
    for (let x = 0; x < w; x += spacing) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
    }
    for (let y = 0; y < h; y += spacing) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }
}

function drawEdges() {
    if (!graph) return;

    const blockedSet = new Set();
    if (envState && envState.graph && envState.graph.edges) {
        envState.graph.edges.forEach(e => {
            if (e.blocked) blockedSet.add(e.from + '->' + e.to);
        });
    }

    graph.edges.forEach(edge => {
        const from = getNodeScreenPos(edge.from);
        const to = getNodeScreenPos(edge.to);
        const key = edge.from + '->' + edge.to;
        const isBlocked = blockedSet.has(key);

        // Calculate control point for curved edges
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const mx = (from.x + to.x) / 2;
        const my = (from.y + to.y) / 2;
        
        // Slight curve offset
        const len = Math.sqrt(dx * dx + dy * dy);
        const nx = -dy / len;
        const ny = dx / len;
        const curveAmount = 15;
        const cx = mx + nx * curveAmount;
        const cy = my + ny * curveAmount;

        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.quadraticCurveTo(cx, cy, to.x, to.y);

        if (isBlocked) {
            ctx.strokeStyle = 'rgba(251, 113, 133, 0.6)';
            ctx.lineWidth = 2.5;
            ctx.setLineDash([6, 4]);
        } else {
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([]);
        }
        ctx.stroke();
        ctx.setLineDash([]);

        // Arrowhead
        const t = 0.7;
        const arrowX = (1 - t) * (1 - t) * from.x + 2 * (1 - t) * t * cx + t * t * to.x;
        const arrowY = (1 - t) * (1 - t) * from.y + 2 * (1 - t) * t * cy + t * t * to.y;

        // Tangent at t
        const tangentX = 2 * (1 - t) * (cx - from.x) + 2 * t * (to.x - cx);
        const tangentY = 2 * (1 - t) * (cy - from.y) + 2 * t * (to.y - cy);
        const angle = Math.atan2(tangentY, tangentX);

        const arrowSize = 7;
        ctx.beginPath();
        ctx.moveTo(arrowX, arrowY);
        ctx.lineTo(
            arrowX - arrowSize * Math.cos(angle - 0.4),
            arrowY - arrowSize * Math.sin(angle - 0.4)
        );
        ctx.moveTo(arrowX, arrowY);
        ctx.lineTo(
            arrowX - arrowSize * Math.cos(angle + 0.4),
            arrowY - arrowSize * Math.sin(angle + 0.4)
        );
        ctx.strokeStyle = isBlocked ? 'rgba(251, 113, 133, 0.6)' : 'rgba(255, 255, 255, 0.15)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Blocked X mark
        if (isBlocked) {
            const markSize = 6;
            ctx.beginPath();
            ctx.moveTo(mx - markSize, my - markSize);
            ctx.lineTo(mx + markSize, my + markSize);
            ctx.moveTo(mx + markSize, my - markSize);
            ctx.lineTo(mx - markSize, my + markSize);
            ctx.strokeStyle = 'rgba(251, 113, 133, 0.8)';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    });
}

function drawNodes() {
    if (!graph) return;

    graph.nodes.forEach(node => {
        const pos = getNodeScreenPos(node.id);
        const color = NODE_COLORS[node.type] || '#ffffff';

        // Glow
        const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, NODE_RADIUS * 2);
        gradient.addColorStop(0, color + '20');
        gradient.addColorStop(1, 'transparent');
        ctx.fillStyle = gradient;
        ctx.fillRect(pos.x - NODE_RADIUS * 2, pos.y - NODE_RADIUS * 2, NODE_RADIUS * 4, NODE_RADIUS * 4);

        // Node circle
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, NODE_RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(17, 24, 39, 0.9)';
        ctx.fill();
        ctx.strokeStyle = color + '80';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Label
        ctx.fillStyle = color;
        ctx.font = '600 11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(node.id, pos.x, pos.y);

        // Type label below
        ctx.fillStyle = 'rgba(255, 255, 255, 0.25)';
        ctx.font = '400 8px Inter, sans-serif';
        ctx.fillText(node.type, pos.x, pos.y + NODE_RADIUS + 10);
    });
}

function drawTrains() {
    if (!envState || !envState.trains) return;

    // Group trains by current position for stacking
    const posGroups = {};
    envState.trains.forEach(train => {
        const key = train.position;
        if (!posGroups[key]) posGroups[key] = [];
        posGroups[key].push(train);
    });

    envState.trains.forEach((train, idx) => {
        const pos = getTrainScreenPos(train);
        const color = TRAIN_COLORS[train.id % TRAIN_COLORS.length];

        // Stack offset for multiple trains at same node
        const group = posGroups[train.position];
        const groupIdx = group.indexOf(train);
        const groupSize = group.length;
        let offsetX = 0;
        let offsetY = 0;
        if (groupSize > 1 && animProgress >= 0.9) {
            const angle = (groupIdx / groupSize) * Math.PI * 2 - Math.PI / 2;
            const stackRadius = NODE_RADIUS + 8;
            offsetX = Math.cos(angle) * stackRadius;
            offsetY = Math.sin(angle) * stackRadius;
        } else if (groupSize <= 1) {
            // Single train: place at node edge
            offsetX = 0;
            offsetY = -NODE_RADIUS - 10;
        }

        const tx = pos.x + offsetX;
        const ty = pos.y + offsetY;

        // Glow for arrived trains
        if (train.status === 'arrived') {
            const glow = ctx.createRadialGradient(tx, ty, 0, tx, ty, TRAIN_RADIUS * 3);
            glow.addColorStop(0, '#34d39940');
            glow.addColorStop(1, 'transparent');
            ctx.fillStyle = glow;
            ctx.beginPath();
            ctx.arc(tx, ty, TRAIN_RADIUS * 3, 0, Math.PI * 2);
            ctx.fill();
        }

        // Train dot
        ctx.beginPath();
        ctx.arc(tx, ty, TRAIN_RADIUS, 0, Math.PI * 2);

        if (train.status === 'arrived') {
            ctx.fillStyle = '#34d399';
        } else if (train.status === 'held') {
            ctx.fillStyle = '#fbbf24';
        } else {
            ctx.fillStyle = color;
        }
        ctx.fill();

        // Border
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Train ID label
        ctx.fillStyle = '#0a0e17';
        ctx.font = '700 9px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(train.id, tx, ty);

        // Delay badge (if any)
        if (train.delay > 0 && train.status !== 'arrived') {
            const badgeX = tx + TRAIN_RADIUS + 3;
            const badgeY = ty - TRAIN_RADIUS - 2;
            ctx.fillStyle = 'rgba(251, 113, 133, 0.9)';
            ctx.font = '600 8px JetBrains Mono, monospace';
            ctx.textAlign = 'left';
            ctx.fillText('+' + train.delay, badgeX, badgeY);
        }
    });
}

// =============================================================
// Animation Loop
// =============================================================

function animationLoop(timestamp) {
    if (animProgress < 1.0) {
        const elapsed = timestamp - animStartTime;
        animProgress = Math.min(1.0, elapsed / ANIM_DURATION);
    }

    render();
    requestAnimationFrame(animationLoop);
}

// =============================================================
// UI Updates
// =============================================================

function updateUI(data) {
    if (!data || !data.state) return;

    const state = data.state;
    const score = data.score != null ? data.score : 0;

    // Score
    const scoreStr = score.toFixed(3);
    scoreValue.textContent = scoreStr;
    scoreFill.style.width = (score * 100) + '%';

    // Color score based on value
    if (score >= 0.8) {
        scoreValue.style.color = '#34d399';
    } else if (score >= 0.5) {
        scoreValue.style.color = '#fbbf24';
    } else {
        scoreValue.style.color = '#fb7185';
    }

    // Stats
    statTimestep.textContent = state.timestep;
    statMaxsteps.textContent = state.max_steps;
    statDelay.textContent = state.total_delay;
    statConflicts.textContent = totalConflicts;

    // Progress
    const progress = state.max_steps > 0
        ? Math.round((state.timestep / state.max_steps) * 100)
        : 0;
    progressFill.style.width = progress + '%';
    progressText.textContent = progress + '%';

    // Train table
    updateTrainTable(state.trains);

    // Timeline
    updateTimeline();

    // Status
    if (state.done) {
        setStatus('done', 'Episode Done');
        btnStep.disabled = true;
        btnAuto.disabled = true;
        stopAuto();
    }

    // Events from info
    if (data.info) {
        if (data.info.conflict_edges) {
            data.info.conflict_edges.forEach(ce => {
                addEvent('conflict',
                    `t=${state.timestep}: Conflict on ${ce.edge[0]}->${ce.edge[1]} | ` +
                    `Winner: T${ce.winner}, Blocked: ${ce.blocked.map(b => 'T'+b).join(', ')}`
                );
            });
        }
        if (data.info.cascade_blocked && data.info.cascade_blocked.length > 0) {
            addEvent('conflict',
                `t=${state.timestep}: Cascade blocked: ${data.info.cascade_blocked.map(b => 'T'+b).join(', ')}`
            );
        }
    }

    // Check for arrivals
    if (state.trains) {
        state.trains.forEach(t => {
            if (t.status === 'arrived' && prevTrainPositions[t.id] && prevTrainPositions[t.id] !== 'T1') {
                addEvent('arrival', `t=${state.timestep}: Train ${t.id} ARRIVED at T1!`);
            }
        });
    }
}

function updateTrainTable(trains) {
    if (!trains) return;

    trainTableBody.innerHTML = '';
    trains.forEach(t => {
        const color = TRAIN_COLORS[t.id % TRAIN_COLORS.length];
        let status, statusClass;

        const tStatus = t.status || '';
        if (tStatus === 'arrived') {
            status = 'Arrived';
            statusClass = 'arrived';
        } else if (tStatus === 'held') {
            status = 'Held';
            statusClass = 'held';
        } else if (tStatus === 'moving') {
            status = 'Moving';
            statusClass = 'moving';
        } else if (tStatus === 'blocked' || tStatus === 'stranded') {
            status = tStatus.charAt(0).toUpperCase() + tStatus.slice(1);
            statusClass = 'blocked';
        } else {
            status = 'Waiting';
            statusClass = 'blocked';
        }

        let delayClass = 'low';
        if (t.delay >= 5) delayClass = 'high';
        else if (t.delay >= 2) delayClass = 'medium';

        const row = document.createElement('tr');
        row.innerHTML = `
            <td>
                <div class="train-id-cell">
                    <span class="train-color-dot" style="background:${color}; box-shadow: 0 0 6px ${color}60"></span>
                    <span class="train-id-label">T${t.id}</span>
                </div>
            </td>
            <td><span class="train-pos">${t.position}</span></td>
            <td><span class="train-delay ${delayClass}">${t.delay}</span></td>
            <td><span class="status-badge ${statusClass}">${status}</span></td>
        `;
        trainTableBody.appendChild(row);
    });
}

function updateTimeline() {
    timelineBars.innerHTML = '';
    if (stepHistory.length === 0) return;

    const maxDelay = Math.max(1, ...stepHistory.map(s => s.delay));
    const maxConflict = Math.max(1, ...stepHistory.map(s => s.conflicts));
    const barHeight = 44; // max height in px

    stepHistory.forEach(s => {
        const delayH = (s.delay / maxDelay) * barHeight * 0.6;
        const conflictH = (s.conflicts / maxConflict) * barHeight * 0.4;

        const bar = document.createElement('div');
        bar.className = 'timeline-bar';
        bar.title = `Step ${s.step}: Delay +${s.delay}, Conflicts ${s.conflicts}`;
        bar.innerHTML = `
            <div class="timeline-bar-delay" style="height:${Math.max(1, delayH)}px"></div>
            <div class="timeline-bar-conflict" style="height:${conflictH}px"></div>
        `;
        timelineBars.appendChild(bar);
    });

    // Scroll to end
    const container = document.getElementById('timeline-container');
    container.scrollLeft = container.scrollWidth;
}

function addEvent(type, message) {
    const entry = document.createElement('div');
    entry.className = `event-entry event-${type}`;
    entry.textContent = message;
    eventLog.insertBefore(entry, eventLog.firstChild);

    // Keep max 50 entries
    while (eventLog.children.length > 50) {
        eventLog.removeChild(eventLog.lastChild);
    }
}

function setStatus(state, text) {
    statusDot.className = 'status-dot ' + state;
    statusText.textContent = text;
}

function clearEventLog() {
    eventLog.innerHTML = '';
}

// =============================================================
// Auto-play
// =============================================================

function startAuto() {
    if (autoInterval) return;
    const speed = parseInt(speedSlider.value);

    btnAuto.style.display = 'none';
    btnPause.style.display = 'flex';
    btnStep.disabled = true;
    setStatus('running', 'Running...');

    autoInterval = setInterval(async () => {
        try {
            const data = await stepEnv();
            updateUI(data);
            if (data.done) {
                stopAuto();
                addEvent('info', `Episode complete! Final score: ${(data.score || 0).toFixed(3)}`);
            }
        } catch (err) {
            stopAuto();
            addEvent('conflict', 'Error: ' + err.message);
        }
    }, speed);
}

function stopAuto() {
    if (autoInterval) {
        clearInterval(autoInterval);
        autoInterval = null;
    }
    btnAuto.style.display = 'flex';
    btnPause.style.display = 'none';
    btnStep.disabled = false;

    if (envState && envState.done) {
        setStatus('done', 'Episode Done');
    } else {
        setStatus('', 'Paused');
    }
}

// =============================================================
// Event Handlers
// =============================================================

btnReset.addEventListener('click', async () => {
    try {
        btnReset.disabled = true;
        stopAuto();
        clearEventLog();

        const task = taskSelect.value;
        const data = await resetEnv(task);

        overlay.classList.add('hidden');
        btnStep.disabled = false;
        btnAuto.disabled = false;

        updateUI(data);
        addEvent('info', `Environment reset with task "${task}" (${data.state.trains.length} trains)`);

        // Log blocked edges
        if (data.state.graph && data.state.graph.edges) {
            const blocked = data.state.graph.edges.filter(e => e.blocked);
            if (blocked.length > 0) {
                addEvent('action',
                    'Blocked: ' + blocked.map(e => e.from + '->' + e.to).join(', ')
                );
            }
        }

        setStatus('', 'Ready');
    } catch (err) {
        addEvent('conflict', 'Reset error: ' + err.message);
    } finally {
        btnReset.disabled = false;
    }
});

btnStep.addEventListener('click', async () => {
    try {
        btnStep.disabled = true;
        const data = await stepEnv();
        updateUI(data);

        if (data.done) {
            addEvent('info', `Episode complete! Final score: ${(data.score || 0).toFixed(3)}`);
        }
    } catch (err) {
        addEvent('conflict', 'Step error: ' + err.message);
    } finally {
        if (envState && !envState.done) {
            btnStep.disabled = false;
        }
    }
});

btnAuto.addEventListener('click', () => startAuto());
btnPause.addEventListener('click', () => stopAuto());

speedSlider.addEventListener('input', () => {
    const val = speedSlider.value;
    speedLabel.textContent = val + 'ms';

    // If auto-playing, restart with new speed
    if (autoInterval) {
        stopAuto();
        startAuto();
    }
});

// =============================================================
// Initialization
// =============================================================

async function init() {
    resizeCanvas();

    try {
        await loadGraph();
        render();
        addEvent('info', 'Graph loaded. Select a task and press Reset.');
    } catch (err) {
        addEvent('conflict', 'Failed to load graph: ' + err.message);
    }

    requestAnimationFrame(animationLoop);
}

init();
