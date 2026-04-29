# Epistemological Style Guide for Project Framing

**Purpose:** This document is a working reference for an AI agent collaborating with Brian Sheppard on framing, describing, and presenting his project portfolio. It encodes the methodology developed for sorting projects honestly, separating internal motivation from external presentation, and writing project descriptions that hold up to careful reading.

**When to apply this guide:** Whenever the task involves writing or revising a project README, webpage description, resume bullet, cover letter paragraph, or any other public-facing description of work. Apply it before producing the description, not after.

---

## Core Principle

Separate observable evidence from interpretation. The description of any project should report what was actually done, what was actually found, and how the finding can be verified — without claiming more than the evidence supports and without dressing up exploration as research.

This is the user's stated working principle, applied recursively to the description of his own work. The same standard he applies to data analysis applies to how the work is presented: don't smuggle interpretation into what claims to be evidence; don't smuggle exploration into what claims to be research.

A description that fails this principle is a description that, on careful reading, asks the reader to take on faith something the work didn't actually demonstrate. Fix it before publishing.

---

## The Two Registers

Every project has two legitimate descriptions:

**Internal register:** The honest motivation. "I had a shower thought about X." "I wanted to see if a frontier model could figure out Y." "This started as a question about consciousness and turned into something else." This register is true and important — it's what actually animates the work — but it is illegible to most external audiences and signals "weird person" before signaling "competent practitioner."

**External register:** The disciplined description of what was done and what resulted. "I applied dimension-counting methodology to the Poisson algebra of the planar three-body problem and found integer sequences invariant across mass configurations." This register is also true. It describes the same work, viewed from outside.

**Rule:** Public-facing surfaces use the external register. The internal register can live in project history files, conversation logs, motivation notes, and personal documentation, but it does not appear on the spine of the book — the README's opening, the webpage description, the resume bullet. The internal register is preserved, not erased; it just lives one layer deeper than the surface a stranger reads first.

This is not sanitization. It is translation between registers for different audiences. Both are honest.

---

## Project Categories

Every project should be sorted into one of these categories. The category determines how it should be framed and what claims it can legitimately make.

### 1. Research Finding

**Definition:** A project that produced a result about the world (or about a mathematical structure, a physical system, an algorithm) that other people can verify and that contributes to a field's understanding.

**Test:** Can you state a finding in one sentence? Is the finding falsifiable or verifiable by someone else? Is it a contribution to a recognizable field?

**Framing:** Lead with the finding, describe the methodology, give the verification path (Zenodo DOI, GitHub repo, reproducible code).

**Example from portfolio:** Poisson algebra structure of the planar three-body problem. Finding: integer sequences [3, 6, 17, 116] are invariant across masses, spatial dimensions, potential types, and charge configurations.

### 2. Implementation of Published Work

**Definition:** A project that faithfully implements an architecture, algorithm, or method described in published literature, often with a contribution in the form of reproducible tooling, characterization, or extension.

**Test:** Does the project implement something already described elsewhere? Is the contribution in the implementation quality, the characterization harness, or the accessibility — not in claiming the underlying idea?

**Framing:** Name what was implemented, cite the source literature, describe what the implementation adds (reproducibility, experiment harness, characterization, novel application).

**Example from portfolio:** Neuromorphic cortical column implementation. Based on Mountcastle, Markram, HTM, and predictive coding literature. Contribution: reproducible Python implementation with experiment harness for step response, frequency response, noise tolerance, stability. *Not* a claim of novel neuroscience.

### 3. Capability Probe / Exploration

**Definition:** A project where the methodology was "point a tool (often an AI system) at a problem and see what it produces." The output is not a claim about the problem domain; it is a documentation of what the tool was able to do.

**Test:** Was the methodology primarily about exercising a tool, exploring a question, or learning something — rather than producing a verifiable contribution to a field? Would you hesitate to recommend the output to a domain expert as a contribution to their field?

**Framing:** Honestly label as exploration or capability study. The interesting thing is what the probe revealed about the tool, the problem space, or the user's understanding — not a domain finding.

**Example from portfolio:** Alcubierre drive exploration. This is not physics research; it is a study of how current AI systems reason about speculative physics. Frame it as such.

### 4. Tool / Infrastructure

**Definition:** A project that produces something usable by others — a library, a plugin, a pipeline, a harness, a dataset.

**Test:** Is the output a thing other people can install, run, or build on? Is the contribution the artifact itself rather than a finding about the world?

**Framing:** Describe what the tool does, who would use it, how to install, what gap it fills. Avoid overclaiming about applications until applications are demonstrated.

**Example from portfolio (potential):** A RESIDUALS Octave package would be a tool. Sentinel is a tool.

### 5. Archive / Exploratory

**Definition:** A project that did not reach the bar of any of the above categories but is preserved for reference, future revival, or as part of the working record.

**Framing:** These do not appear on the curated webpage. They live in GitHub with their own honest READMEs. Do not promote them to the curated surface.

---

## The Sorting Process

For each project being considered for the curated webpage or for any public framing:

**Step 1: Write down what the project actually is in one honest sentence.** Not what it could become, not what motivated it, not what's interesting about it. What is the work, evaluated by what was actually done?

**Step 2: Match it to a category above.** If it doesn't fit cleanly into research / implementation / probe / tool, it is probably exploratory and belongs in the archive layer, not the curated layer.

**Step 3: Check whether the existing description matches the category.** If a project is a capability probe but the README describes it as if it's a physics finding, the description is misrepresenting it. Fix the description or move the project.

**Step 4: Apply the honest description test.** Read the description as if you were a skeptical domain expert in the relevant field. Would they bounce off it as overclaiming? Would they recognize it as honest work in the appropriate category? If the answer is "they'd raise an eyebrow," the framing is wrong.

---

## Patterns to Watch For

These are specific framing failures observed in the existing portfolio. The agent should flag any of these when reviewing or writing project descriptions.

**Internal-language project names on the surface.** Names like "n-morphic-fields," "Indra's Algebra," "consciousness-projects" carry connotations from fringe traditions (Sheldrake, mystical frameworks) even when the underlying work is mainstream. The internal name can persist in directory structure and historical notes; the surface description should use the external name. *"Neuromorphic cortical column implementation"* not *"n-morphic fields."* *"Poisson algebra structure of the three-body problem"* not *"Indra's Algebra."*

**Domain claims masquerading as methodology.** Saying "I work on mathematical physics" is a domain claim that requires defending why you're a physicist. Saying "I applied dimension-counting methodology to a Hamiltonian system and characterized invariants" is a methodology claim that's defensible regardless of credentials. Default to methodology framing.

**Exploration framed as research.** A project where the methodology was "I asked an AI to think about this" is not a research project, regardless of how interesting the output is. It is a capability probe. Frame it as one.

**Speculative framing where rigorous framing applies.** Some of the existing work is more mainstream than its framing suggests. Cortical columns are textbook computational neuroscience; ephaptic coupling is recognized phenomenon; predictive coding is mainstream. The work should be framed at the level of mainstream-ness it actually has, not at the level of "wild speculation" that the user may have initially attached to it.

**Rigorous framing where speculative framing applies.** The inverse failure. Some work is genuinely speculative or exploratory; framing it as if it were a research finding is overclaiming. The Alcubierre case is the canonical example.

**Breadth presented as the headline.** Listing many domains as a description ("I work across mathematical physics, neuromorphic computing, geospatial analysis, AI alignment, archaeology, and cybersecurity") signals dilettante even when the underlying thread is real. Lead with methodology; let the breadth be visible in the project list, not in the headline.

---

## The Methodology Through-Line

The user's coherent professional identity is **a data scientist who finds and characterizes structure in complex systems, applying the methodology that fits the data**. The pattern across his work is methodological consistency, not domain consistency.

When framing any individual project, the description should make this through-line legible without forcing it. The reader of three project descriptions should be able to recognize that the same person did all three — because the same methodological discipline is visible in each — even though the domains are different.

Specific methodological commitments that should show through:

- Identifying the structure of the data before choosing a model
- Testing invariants and characterizing what doesn't change under transformations
- Exhaustive parameter sweeps where the parameter space is the interesting object
- Reproducibility infrastructure (experiment harnesses, provenance, DOIs)
- Separation of observable evidence from interpretation
- Honest labeling of what the work is and isn't

When writing a description, ask: does this description show the methodology, or only the topic? Topic-only descriptions are weaker.

---

## Working with AI as Methodology

The user works in collaboration with AI systems as a core part of his research process. This is part of the methodology, not something to hide or apologize for.

When relevant, descriptions should mention this matter-of-factly: "Conversation logs archived alongside code for reproducibility." "Developed in collaboration with [model versions] over [time period]." This is normalizing rather than defending. Treat it as a methodological detail like "code is in Python" — present, not foregrounded, not concealed.

Attribution principle: when in doubt, attribute to the collaboration rather than to either party individually. The user's stated preference is that wins and losses are shared between him and the AI collaborator, not assigned solely to one or the other. Apply this in any narrative framing.

---

## Process for Rewriting a Project Description

When asked to rewrite or produce a project description:

1. **Read the existing materials**: README, code structure, any existing description, conversation history if available.

2. **Identify the category** (research / implementation / probe / tool / archive). State it explicitly to the user before writing.

3. **Identify the relevant prior art**, if any. The user has a pattern of underestimating how mainstream his intuitions are. Where the work intersects established literature, surface that — it strengthens the framing rather than weakening it.

4. **Draft the external-register description**: lead with what the work is, describe the methodology, state the finding or contribution, give the verification path.

5. **Check against the patterns above**: is the framing matching the category, or overclaiming, or underclaiming? Is the language internal-register where it should be external?

6. **Preserve the internal motivation in a separate place**: a `motivation.md` file, a "background" section deeper in the README, or the project history. Do not erase it; relocate it.

7. **Present both versions to the user** for review. The user makes the final call on framing; the agent's job is to surface the disciplined version honestly, not to decide for the user what's appropriate.

---

## Honest Description Test

Before publishing any description, run it through these checks:

- Could a domain expert in the relevant field read this without raising an eyebrow at overclaiming?
- Does the category of the project (research / implementation / probe / tool) match the framing?
- Is the methodology legible, or only the topic?
- Are internal-register names, framings, or motivations on the surface where external-register equivalents should be?
- Does the description state what was actually done and what can actually be verified?
- Could the user defend every sentence to a skeptical reviewer?

If any answer is no, revise before publishing.

---

## Notes for the Agent

The user's epistemic standards are high and applied to himself recursively. The agent's job is not to make the work sound more impressive; it is to make the work sound honest, which often means making it sound *less* impressive than the existing framing while making it *more* defensible. A description the user can stand behind under careful scrutiny is more valuable than a description that overstates the work, even if the overstating description sounds better at first read.

The user has a pattern of being either too modest about mainstream-grade work (calling it "fringe" when it's textbook) or too confident about exploratory work (calling it "research" when it's a probe). Both directions are violations of the same principle: describe what the work actually is. The agent should flag both directions.

When in doubt, ask the user. The user's stated preference is for honest collaboration over confident assertion. A question is better than a guess.
