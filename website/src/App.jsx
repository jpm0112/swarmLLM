import { useEffect, useRef, useState } from 'react'
import * as AsciinemaPlayer from 'asciinema-player'
import 'asciinema-player/dist/bundle/asciinema-player.css'
import './App.css'

// ── NAV ──────────────────────────────────────────────────────────────────────
function Nav() {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 60)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <nav className={`nav ${scrolled ? 'nav--scrolled' : ''}`}>
      <div className="nav__logo">
        <span className="nav__logo-swarm">SWARM</span>
        <span className="nav__logo-llm">LLM</span>
      </div>
      <div className="nav__links">
        <a href="#how-it-works">How It Works</a>
        <a href="#features">Features</a>
        <a href="#architecture">Architecture</a>
        <a href="#getting-started">Get Started</a>
        <a href="#demo">Demo</a>
        <a href="https://github.com/jpm0112/swarmLLM" target="_blank" rel="noreferrer" className="nav__github">
          GitHub ↗
        </a>
      </div>
    </nav>
  )
}

// ── HERO CANVAS (swarm network animation) ────────────────────────────────────
function HeroCanvas() {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    let animId

    const resize = () => {
      canvas.width  = canvas.offsetWidth
      canvas.height = canvas.offsetHeight
    }
    resize()
    window.addEventListener('resize', resize, { passive: true })

    // Build nodes: index 0 = coordinator (cyan), rest = workers (amber)
    const NODE_COUNT    = 65
    const CONNECT_DIST  = 130

    const nodes = Array.from({ length: NODE_COUNT }, (_, i) => ({
      x:          Math.random() * canvas.width,
      y:          Math.random() * canvas.height,
      vx:         (Math.random() - 0.5) * 0.35,
      vy:         (Math.random() - 0.5) * 0.35,
      r:          Math.random() * 1.8 + 1.2,
      isCoord:    i === 0,
      phase:      Math.random() * Math.PI * 2,
      phaseSpeed: 0.018 + Math.random() * 0.018,
    }))

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Move nodes
      nodes.forEach(n => {
        n.x += n.vx
        n.y += n.vy
        n.phase += n.phaseSpeed
        if (n.x < 0 || n.x > canvas.width)  n.vx *= -1
        if (n.y < 0 || n.y > canvas.height) n.vy *= -1
      })

      // Draw edges
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx   = nodes[i].x - nodes[j].x
          const dy   = nodes[i].y - nodes[j].y
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < CONNECT_DIST) {
            const t       = 1 - dist / CONNECT_DIST
            const isHot   = nodes[i].isCoord || nodes[j].isCoord
            const alpha   = isHot ? t * 0.35 : t * 0.14
            ctx.strokeStyle = isHot
              ? `rgba(0,212,255,${alpha})`
              : `rgba(255,123,47,${alpha})`
            ctx.lineWidth = 0.6
            ctx.beginPath()
            ctx.moveTo(nodes[i].x, nodes[i].y)
            ctx.lineTo(nodes[j].x, nodes[j].y)
            ctx.stroke()
          }
        }
      }

      // Draw nodes
      nodes.forEach(n => {
        const bright = 0.65 + 0.35 * Math.sin(n.phase)
        if (n.isCoord) {
          ctx.shadowBlur  = 16
          ctx.shadowColor = '#00d4ff'
          ctx.fillStyle   = `rgba(0,212,255,${bright})`
          ctx.beginPath()
          ctx.arc(n.x, n.y, n.r * 2, 0, Math.PI * 2)
          ctx.fill()
        } else {
          ctx.shadowBlur  = 6
          ctx.shadowColor = '#ff7b2f'
          ctx.fillStyle   = `rgba(255,123,47,${bright * 0.75})`
          ctx.beginPath()
          ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2)
          ctx.fill()
        }
        ctx.shadowBlur = 0
      })

      animId = requestAnimationFrame(draw)
    }

    draw()
    return () => {
      cancelAnimationFrame(animId)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return <canvas ref={canvasRef} className="hero__canvas" />
}

// ── HERO ─────────────────────────────────────────────────────────────────────
function Hero() {
  return (
    <section className="hero" id="hero">
      <HeroCanvas />
      <div className="hero__scanlines" />
      <div className="hero__vignette" />
      <div className="hero__content">
        <span className="hero__badge">Research Prototype · Work in Progress</span>
        <h1 className="hero__title">
          <span className="hero__title-swarm">SWARM</span>
          <span className="hero__title-llm">LLM</span>
        </h1>
        <p className="hero__tagline">
          Coordinator-guided LLM swarms<br />for combinatorial optimization
        </p>
        <div className="hero__ctas">
          <a href="https://github.com/jpm0112/swarmLLM" target="_blank" rel="noreferrer" className="btn btn--primary">
            View on GitHub ↗
          </a>
          <a href="#how-it-works" className="btn btn--ghost">
            Explore ↓
          </a>
        </div>
      </div>
      <span className="hero__scroll-hint">scroll to explore ↓</span>
    </section>
  )
}

// ── PITCH ─────────────────────────────────────────────────────────────────────
function Pitch() {
  return (
    <section className="pitch">
      <div className="container">
        <p className="pitch__lead">
          A <em>central coordinator LLM</em> assigns targeted exploration directions to{' '}
          <em>parallel worker agents</em>. Each agent generates a Python heuristic,
          executes it in a sandboxed subprocess, and writes results to a{' '}
          <em>shared markdown log</em> — which the coordinator reads to iteratively
          refine strategy across unlimited rounds.
        </p>
        <div className="pitch__stats">
          <div className="stat">
            <span className="stat__number">N</span>
            <span className="stat__label">parallel<br />worker agents</span>
          </div>
          <div className="stat">
            <span className="stat__number">∞</span>
            <span className="stat__label">iterative<br />refinement loops</span>
          </div>
          <div className="stat">
            <span className="stat__number">2+</span>
            <span className="stat__label">optimization<br />problems</span>
          </div>
          <div className="stat">
            <span className="stat__number">0</span>
            <span className="stat__label">external orchestration<br />dependencies</span>
          </div>
        </div>
      </div>
    </section>
  )
}

// ── HOW IT WORKS ─────────────────────────────────────────────────────────────
const STEPS = [
  {
    num: '01',
    title: 'Problem Definition',
    desc: 'Define an optimization target — job scheduling, job shop scheduling, or bring your own. SwarmLLM handles the evaluation harness and scoring.',
    accent: 'cyan',
  },
  {
    num: '02',
    title: 'Coordinator Assigns Directions',
    desc: 'The coordinator LLM reads the full history of prior attempts and assigns distinct, non-redundant exploration strategies to each worker.',
    accent: 'amber',
  },
  {
    num: '03',
    title: 'Workers Generate Heuristics',
    desc: 'Each worker LLM independently generates a Python heuristic function tailored to its assigned strategy — different approaches in parallel.',
    accent: 'cyan',
  },
  {
    num: '04',
    title: 'Sandboxed Execution',
    desc: 'Agent-generated code runs in an isolated subprocess with a timeout. Solutions are scored against the objective; failures are handled gracefully.',
    accent: 'amber',
  },
  {
    num: '05',
    title: 'Shared Log Update',
    desc: 'Results, scores, and reasoning are appended to a shared markdown log — structured to be readable by both humans and LLMs.',
    accent: 'cyan',
  },
  {
    num: '06',
    title: 'Iterative Refinement',
    desc: 'The coordinator reads the updated log and refines strategy for the next round. The swarm converges toward better solutions over iterations.',
    accent: 'amber',
  },
]

function HowItWorks() {
  return (
    <section className="how-it-works" id="how-it-works">
      <div className="container">
        <div className="section-label">How It Works</div>
        <h2 className="section-title">Six steps.<br />Infinite iterations.</h2>
        <div className="steps">
          {STEPS.map(s => (
            <div key={s.num} className={`step step--${s.accent}`}>
              <span className="step__num">{s.num}</span>
              <h3 className="step__title">{s.title}</h3>
              <p className="step__desc">{s.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

// ── FEATURES ─────────────────────────────────────────────────────────────────
const FEATURES = [
  {
    icon: '◈',
    title: 'Coordinator-Guided Exploration',
    desc: 'The coordinator reads the full history and assigns targeted, differentiated strategies — preventing redundant parallel exploration.',
    tag: 'Core',
  },
  {
    icon: '⬡',
    title: 'Sandboxed Code Execution',
    desc: 'Agent-generated Python heuristics run in isolated subprocesses. Faulty code fails safely; execution timeout enforced.',
    tag: 'Safety',
  },
  {
    icon: '⟐',
    title: 'Inspectable Shared Logs',
    desc: 'All results and agent reasoning are stored in plain markdown — readable by humans for debugging and by LLMs for context.',
    tag: 'Transparency',
  },
  {
    icon: '⊕',
    title: 'Flexible Inference Backends',
    desc: 'Unified OpenAI-compatible interface supports Ollama, vLLM (Metal/Linux), and MLX-LM. Swap backends with a single CLI flag.',
    tag: 'Infrastructure',
  },
  {
    icon: '⊞',
    title: 'Rich TUI Dashboard',
    desc: 'Real-time terminal UI with live agent status, per-iteration scores, token tracking, and full telemetry — powered by Rich.',
    tag: 'Observability',
  },
  {
    icon: '⊛',
    title: 'Extensible Problem Framework',
    desc: 'Add new optimization problems with a problem definition class and a prompt template. Zero changes to the orchestration core.',
    tag: 'Extensibility',
  },
]

function Features() {
  return (
    <section className="features" id="features">
      <div className="container">
        <div className="section-label">Capabilities</div>
        <h2 className="section-title">Built for<br />serious experimentation.</h2>
        <div className="features__grid">
          {FEATURES.map(f => (
            <div key={f.title} className="feature-card">
              <div className="feature-card__header">
                <span className="feature-card__icon">{f.icon}</span>
                <span className="feature-card__tag">{f.tag}</span>
              </div>
              <h3 className="feature-card__title">{f.title}</h3>
              <p className="feature-card__desc">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

// ── ARCHITECTURE ─────────────────────────────────────────────────────────────
function Architecture() {
  return (
    <section className="architecture" id="architecture">
      <div className="container">
        <div className="section-label">System Design</div>
        <h2 className="section-title">Architecture</h2>

        <div className="arch-diagram">
          {/* Problem */}
          <div className="arch-row">
            <div className="arch-node arch-node--problem">
              <span className="arch-node__label">Problem</span>
              <span className="arch-node__sub">job scheduling · JSP · extensible</span>
            </div>
          </div>

          <div className="arch-connector">↓</div>

          {/* Coordinator */}
          <div className="arch-row">
            <div className="arch-node arch-node--coordinator">
              <span className="arch-node__label">Coordinator LLM</span>
              <span className="arch-node__sub">reads logs · assigns exploration directions</span>
            </div>
          </div>

          <div className="arch-connector" style={{ letterSpacing: '10px' }}>↙ ↓ ↘</div>

          {/* Workers */}
          <div className="arch-row arch-row--workers">
            {[
              { label: 'Worker Agent 1', sub: 'generates heuristic' },
              { label: 'Worker Agent 2', sub: 'generates heuristic' },
              { label: 'Worker Agent N', sub: 'generates heuristic', extra: 'arch-node--worker-n' },
            ].map(w => (
              <div key={w.label} className={`arch-node arch-node--worker ${w.extra ?? ''}`}>
                <span className="arch-node__label">{w.label}</span>
                <span className="arch-node__sub">{w.sub}</span>
              </div>
            ))}
          </div>

          <div className="arch-connector" style={{ letterSpacing: '10px' }}>↘ ↓ ↙</div>

          {/* Sandbox */}
          <div className="arch-row">
            <div className="arch-node arch-node--sandbox">
              <span className="arch-node__label">Sandbox</span>
              <span className="arch-node__sub">isolated subprocess · scored execution</span>
            </div>
          </div>

          <div className="arch-connector">↓</div>

          {/* Shared Logs */}
          <div className="arch-row">
            <div className="arch-node arch-node--logs">
              <span className="arch-node__label">Shared Logs</span>
              <span className="arch-node__sub">markdown · scores · reasoning · history</span>
            </div>
          </div>

          {/* Feedback loop label */}
          <div className="arch-feedback">
            <span className="arch-feedback__arrow">↺</span>
            <span className="arch-feedback__label">
              logs feed back into coordinator · loop repeats until convergence
            </span>
          </div>
        </div>

        <div className="tech-stack">
          <span className="tech-stack__label">Built with</span>
          {['Python', 'PydanticAI', 'Gurobi', 'OR-Tools', 'Rich', 'NumPy', 'NetworkX'].map(t => (
            <span key={t} className="tech-tag">{t}</span>
          ))}
        </div>
      </div>
    </section>
  )
}

// ── GETTING STARTED ───────────────────────────────────────────────────────────
function GettingStarted() {
  const [copied, setCopied] = useState(false)

  const raw = [
    '# clone and install',
    'git clone https://github.com/jpm0112/swarmLLM',
    'cd swarmLLM',
    'uv pip install -e .',
    '',
    '# run a swarm on job scheduling',
    'python scripts/run.py \\',
    '  --problem job_scheduling \\',
    '  --agents 4 \\',
    '  --iterations 10',
  ].join('\n')

  const copy = () => {
    navigator.clipboard.writeText(raw)
    setCopied(true)
    setTimeout(() => setCopied(false), 2200)
  }

  return (
    <section className="getting-started" id="getting-started">
      <div className="container">
        <div className="section-label">Quickstart</div>
        <h2 className="section-title">Up and running<br />in four commands.</h2>

        <div className="code-block">
          <div className="code-block__header">
            <div className="code-block__dots">
              <span /><span /><span />
            </div>
            <span className="code-block__title">terminal</span>
            <button className="code-block__copy" onClick={copy}>
              {copied ? '✓ copied' : 'copy'}
            </button>
          </div>
          <pre className="code-block__body"><code>
<span className="c-comment"># clone and install</span>{'\n'}
<span className="c-cmd">git clone</span> <span className="c-string">https://github.com/jpm0112/swarmLLM</span>{'\n'}
<span className="c-cmd">cd</span> swarmLLM{'\n'}
<span className="c-cmd">uv pip install</span> -e .{'\n'}
{'\n'}
<span className="c-comment"># run a swarm on job scheduling</span>{'\n'}
<span className="c-cmd">python</span> scripts/run.py \{'\n'}
{'  '}<span className="c-flag">--problem</span> job_scheduling \{'\n'}
{'  '}<span className="c-flag">--agents</span> 4 \{'\n'}
{'  '}<span className="c-flag">--iterations</span> 10
          </code></pre>
        </div>

        <div className="getting-started__prereqs">
          <div className="prereq">
            <span className="prereq__dot" />
            Requires Python 3.11+ and <strong>uv</strong>
          </div>
          <div className="prereq">
            <span className="prereq__dot" />
            Default backend: <strong>Ollama</strong> (local)
          </div>
          <div className="prereq">
            <span className="prereq__dot" />
            Also supports vLLM, MLX-LM via <code>--backend</code>
          </div>
        </div>
      </div>
    </section>
  )
}

// ── DEMO ─────────────────────────────────────────────────────────────────────
function Demo() {
  const playerRef = useRef(null)

  useEffect(() => {
    const player = AsciinemaPlayer.create('/demo.cast', playerRef.current, {
      fit: 'width',
      autoPlay: false,
      controls: true,
      speed: 1.5,
      idleTimeLimit: 1,
      theme: 'monokai',
    })
    return () => player.dispose()
  }, [])

  return (
    <section className="demo" id="demo">
      <div className="container">
        <div className="section-label">Live Demo</div>
        <h2 className="section-title">See it run.</h2>
        <div className="demo__player" ref={playerRef} />
        <p className="demo__caption">
          Setup + full TUI run on <strong>ollama-cloud</strong> · recorded with asciinema
        </p>
      </div>
    </section>
  )
}

// ── TEAM ─────────────────────────────────────────────────────────────────────
function Team() {
  return (
    <section className="team" id="team">
      <div className="container">
        <div className="section-label">The Team</div>
        <h2 className="section-title">Equal contribution.</h2>

        <div className="team__members">
          <div className="team-member">
            <div className="team-member__avatar">KZ</div>
            <h3 className="team-member__name">Konstantinos Ziliaskopoulos</h3>
            <span className="team-member__contrib">★ Equal Contribution</span>
          </div>

          <div className="team__divider">+</div>

          <div className="team-member">
            <div className="team-member__avatar">JPM</div>
            <h3 className="team-member__name">Juan Pablo Morande</h3>
            <span className="team-member__contrib">★ Equal Contribution</span>
          </div>
        </div>
      </div>
    </section>
  )
}

// ── FOOTER ────────────────────────────────────────────────────────────────────
function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer__cta">
          <h2 className="footer__cta-title">
            Open source.<br /><em>Work in progress.</em>
          </h2>
          <p className="footer__cta-desc">
            SwarmLLM is a research prototype under active development.
            Contributions, experiments, and feedback are welcome.
          </p>
          <a
            href="https://github.com/jpm0112/swarmLLM"
            target="_blank"
            rel="noreferrer"
            className="btn btn--primary btn--large"
          >
            View on GitHub ↗
          </a>
        </div>

        <div className="footer__bottom">
          <span className="footer__logo">
            <span className="nav__logo-swarm">SWARM</span>
            <span className="nav__logo-llm">LLM</span>
          </span>
          <span className="footer__copy">
            Ziliaskopoulos · Morande · Research Prototype
          </span>
        </div>
      </div>
    </footer>
  )
}

// ── APP ───────────────────────────────────────────────────────────────────────
export default function App() {
  return (
    <>
      <Nav />
      <Hero />
      <Pitch />
      <HowItWorks />
      <Features />
      <Architecture />
      <GettingStarted />
      <Demo />
      <Team />
      <Footer />
    </>
  )
}
