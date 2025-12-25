/**
 * AIController.js - Intelligent agent controller for the 3D world
 * Manages AI decision-making, pathfinding, and behavior for NPCs
 */

export class AIController {
    constructor(config = {}) {
        this.config = {
            updateInterval: config.updateInterval || 100, // ms
            perceptionRadius: config.perceptionRadius || 20,
            memorySize: config.memorySize || 50,
            decisionThreshold: config.decisionThreshold || 0.6,
            useLLM: config.useLLM !== false,
            model: config.model || 'gpt-oss-20',
            ...config
        };

        this.agents = new Map();
        this.worldState = null;
        this.running = false;
        this.updateTimer = null;
        this.llmQueue = [];
        this.processingLLM = false;
    }

    /**
     * Register an agent with the controller
     */
    registerAgent(agent) {
        const agentData = {
            id: agent.id || `agent_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            object: agent.object, // Three.js mesh
            type: agent.type || 'npc',
            personality: agent.personality || 'neutral',
            goals: agent.goals || [],
            memory: [],
            state: {
                position: agent.object.position.clone(),
                velocity: { x: 0, y: 0, z: 0 },
                health: agent.health || 100,
                energy: agent.energy || 100,
                mood: agent.mood || 'neutral'
            },
            perception: {
                nearbyAgents: [],
                nearbyObjects: [],
                threats: [],
                opportunities: []
            },
            currentAction: null,
            lastDecision: Date.now(),
            decisionHistory: []
        };

        this.agents.set(agentData.id, agentData);
        return agentData.id;
    }

    /**
     * Remove an agent from the controller
     */
    unregisterAgent(agentId) {
        return this.agents.delete(agentId);
    }

    /**
     * Set the world state reference
     */
    setWorldState(worldState) {
        this.worldState = worldState;
    }

    /**
     * Start the AI controller update loop
     */
    start() {
        if (this.running) return;
        this.running = true;
        this.updateLoop();
    }

    /**
     * Stop the AI controller
     */
    stop() {
        this.running = false;
        if (this.updateTimer) {
            clearTimeout(this.updateTimer);
            this.updateTimer = null;
        }
    }

    /**
     * Main update loop
     */
    updateLoop() {
        if (!this.running) return;

        // Update all agents
        for (const [agentId, agent] of this.agents) {
            this.updateAgent(agent);
        }

        // Process LLM queue if not already processing
        if (this.config.useLLM && !this.processingLLM && this.llmQueue.length > 0) {
            this.processLLMQueue();
        }

        // Schedule next update
        this.updateTimer = setTimeout(() => this.updateLoop(), this.config.updateInterval);
    }

    /**
     * Update a single agent
     */
    updateAgent(agent) {
        // Update perception
        this.updatePerception(agent);

        // Make decisions based on current state
        const timeSinceLastDecision = Date.now() - agent.lastDecision;
        if (timeSinceLastDecision > this.config.updateInterval * 5) {
            this.makeDecision(agent);
        }

        // Execute current action
        if (agent.currentAction) {
            this.executeAction(agent, agent.currentAction);
        }

        // Update state
        this.updateState(agent);

        // Store in memory
        this.updateMemory(agent);
    }

    /**
     * Update agent's perception of the world
     */
    updatePerception(agent) {
        agent.perception.nearbyAgents = [];
        agent.perception.nearbyObjects = [];
        agent.perception.threats = [];
        agent.perception.opportunities = [];

        const agentPos = agent.object.position;

        // Find nearby agents
        for (const [otherId, otherAgent] of this.agents) {
            if (otherId === agent.id) continue;

            const distance = agentPos.distanceTo(otherAgent.object.position);
            if (distance <= this.config.perceptionRadius) {
                agent.perception.nearbyAgents.push({
                    id: otherId,
                    distance,
                    type: otherAgent.type,
                    position: otherAgent.object.position.clone(),
                    state: { ...otherAgent.state }
                });

                // Identify threats
                if (this.isThreat(agent, otherAgent)) {
                    agent.perception.threats.push(otherId);
                }
            }
        }

        // Find nearby objects from world state
        if (this.worldState && this.worldState.objects) {
            for (const obj of this.worldState.objects) {
                const distance = agentPos.distanceTo(obj.position);
                if (distance <= this.config.perceptionRadius) {
                    agent.perception.nearbyObjects.push({
                        type: obj.type,
                        distance,
                        position: obj.position.clone()
                    });

                    // Identify opportunities
                    if (this.isOpportunity(agent, obj)) {
                        agent.perception.opportunities.push(obj);
                    }
                }
            }
        }
    }

    /**
     * Make a decision for the agent
     */
    makeDecision(agent) {
        agent.lastDecision = Date.now();

        // Rule-based decision making
        const decision = this.ruleBasedDecision(agent);

        if (decision.confidence >= this.config.decisionThreshold) {
            // High confidence, execute immediately
            agent.currentAction = decision.action;
            agent.decisionHistory.push({
                time: Date.now(),
                action: decision.action,
                reason: decision.reason,
                confidence: decision.confidence,
                method: 'rule-based'
            });
        } else if (this.config.useLLM) {
            // Low confidence, queue for LLM decision
            this.queueLLMDecision(agent, decision);
        } else {
            // No LLM, use rule-based decision anyway
            agent.currentAction = decision.action;
            agent.decisionHistory.push({
                time: Date.now(),
                action: decision.action,
                reason: decision.reason,
                confidence: decision.confidence,
                method: 'rule-based-fallback'
            });
        }
    }

    /**
     * Rule-based decision making
     */
    ruleBasedDecision(agent) {
        // Priority system
        const rules = [];

        // Survival rules (high priority)
        if (agent.state.health < 30) {
            rules.push({
                action: { type: 'flee', target: this.findSafeLocation(agent) },
                reason: 'Low health, seeking safety',
                confidence: 0.9,
                priority: 10
            });
        }

        if (agent.state.energy < 20) {
            rules.push({
                action: { type: 'rest', duration: 5000 },
                reason: 'Low energy, need to rest',
                confidence: 0.85,
                priority: 9
            });
        }

        // Threat response (high priority)
        if (agent.perception.threats.length > 0) {
            const threat = agent.perception.nearbyAgents.find(a => 
                agent.perception.threats.includes(a.id)
            );
            if (threat && threat.distance < 10) {
                rules.push({
                    action: { type: 'flee', target: this.calculateFleeTarget(agent, threat) },
                    reason: 'Immediate threat detected',
                    confidence: 0.95,
                    priority: 10
                });
            }
        }

        // Goal-oriented behavior (medium priority)
        if (agent.goals.length > 0) {
            const currentGoal = agent.goals[0];
            rules.push({
                action: { type: 'pursue_goal', goal: currentGoal },
                reason: `Working towards goal: ${currentGoal.description}`,
                confidence: 0.7,
                priority: 5
            });
        }

        // Opportunity seeking (medium priority)
        if (agent.perception.opportunities.length > 0) {
            const opportunity = agent.perception.opportunities[0];
            rules.push({
                action: { type: 'investigate', target: opportunity.position },
                reason: 'Opportunity detected',
                confidence: 0.6,
                priority: 4
            });
        }

        // Social behavior (low priority)
        if (agent.personality === 'friendly' && agent.perception.nearbyAgents.length > 0) {
            const nearby = agent.perception.nearbyAgents[0];
            rules.push({
                action: { type: 'approach', target: nearby.position },
                reason: 'Social interaction opportunity',
                confidence: 0.5,
                priority: 3
            });
        }

        // Default behavior (lowest priority)
        rules.push({
            action: { type: 'wander' },
            reason: 'No specific stimulus, exploring',
            confidence: 0.4,
            priority: 1
        });

        // Sort by priority and return highest
        rules.sort((a, b) => b.priority - a.priority);
        return rules[0];
    }

    /**
     * Queue a decision for LLM processing
     */
    queueLLMDecision(agent, fallbackDecision) {
        this.llmQueue.push({
            agent,
            fallbackDecision,
            timestamp: Date.now()
        });
    }

    /**
     * Process the LLM decision queue
     */
    async processLLMQueue() {
        if (this.processingLLM || this.llmQueue.length === 0) return;

        this.processingLLM = true;
        const item = this.llmQueue.shift();
        const { agent, fallbackDecision } = item;

        try {
            // Create context for LLM
            const context = this.createLLMContext(agent);
            
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: [
                        {
                            role: 'system',
                            content: 'You are an AI agent controller. Analyze the situation and decide the best action. Respond with JSON: {action: {type: string, ...params}, reason: string, confidence: number}'
                        },
                        {
                            role: 'user',
                            content: context
                        }
                    ],
                    model: this.config.model
                })
            });

            const data = await response.json();
            const decision = this.parseLLMDecision(data.text, fallbackDecision);

            agent.currentAction = decision.action;
            agent.decisionHistory.push({
                time: Date.now(),
                action: decision.action,
                reason: decision.reason,
                confidence: decision.confidence,
                method: 'llm'
            });

        } catch (error) {
            console.error('LLM decision error:', error);
            // Use fallback decision
            agent.currentAction = fallbackDecision.action;
            agent.decisionHistory.push({
                time: Date.now(),
                action: fallbackDecision.action,
                reason: fallbackDecision.reason + ' (LLM failed)',
                confidence: fallbackDecision.confidence,
                method: 'fallback'
            });
        } finally {
            this.processingLLM = false;
        }
    }

    /**
     * Create context description for LLM
     */
    createLLMContext(agent) {
        return `
Agent Status:
- Type: ${agent.type}
- Personality: ${agent.personality}
- Health: ${agent.state.health}
- Energy: ${agent.state.energy}
- Mood: ${agent.state.mood}
- Position: (${agent.state.position.x.toFixed(1)}, ${agent.state.position.y.toFixed(1)}, ${agent.state.position.z.toFixed(1)})

Perception:
- Nearby agents: ${agent.perception.nearbyAgents.length}
- Nearby objects: ${agent.perception.nearbyObjects.length}
- Threats: ${agent.perception.threats.length}
- Opportunities: ${agent.perception.opportunities.length}

Goals: ${agent.goals.map(g => g.description).join(', ') || 'None'}

Recent actions: ${agent.decisionHistory.slice(-3).map(d => d.action.type).join(', ')}

What should this agent do next?
        `.trim();
    }

    /**
     * Parse LLM response into decision
     */
    parseLLMDecision(text, fallback) {
        try {
            // Try to extract JSON from response
            const jsonMatch = text.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
                const decision = JSON.parse(jsonMatch[0]);
                return {
                    action: decision.action,
                    reason: decision.reason,
                    confidence: decision.confidence || 0.8
                };
            }
        } catch (error) {
            console.error('Failed to parse LLM decision:', error);
        }
        return fallback;
    }

    /**
     * Execute an action
     */
    executeAction(agent, action) {
        switch (action.type) {
            case 'move':
                this.moveTowards(agent, action.target, action.speed || 0.1);
                break;
            case 'flee':
                this.moveTowards(agent, action.target, action.speed || 0.2);
                break;
            case 'approach':
                this.moveTowards(agent, action.target, action.speed || 0.05);
                break;
            case 'wander':
                this.wander(agent);
                break;
            case 'rest':
                agent.state.energy = Math.min(100, agent.state.energy + 0.5);
                break;
            case 'investigate':
                this.moveTowards(agent, action.target, 0.08);
                break;
            case 'pursue_goal':
                this.pursueGoal(agent, action.goal);
                break;
            default:
                console.warn(`Unknown action type: ${action.type}`);
        }
    }

    /**
     * Move agent towards a target
     */
    moveTowards(agent, target, speed) {
        if (!target) return;

        const direction = new THREE.Vector3(
            target.x - agent.object.position.x,
            0,
            target.z - agent.object.position.z
        ).normalize();

        agent.object.position.x += direction.x * speed;
        agent.object.position.z += direction.z * speed;

        agent.state.velocity = { x: direction.x * speed, y: 0, z: direction.z * speed };
        agent.state.energy = Math.max(0, agent.state.energy - 0.1);
    }

    /**
     * Random wandering behavior
     */
    wander(agent) {
        if (!agent.wanderTarget || Math.random() < 0.05) {
            // Set new random target
            agent.wanderTarget = {
                x: agent.object.position.x + (Math.random() - 0.5) * 20,
                z: agent.object.position.z + (Math.random() - 0.5) * 20
            };
        }
        this.moveTowards(agent, agent.wanderTarget, 0.05);
    }

    /**
     * Pursue a goal
     */
    pursueGoal(agent, goal) {
        if (goal.location) {
            this.moveTowards(agent, goal.location, 0.1);
            
            // Check if goal reached
            const distance = agent.object.position.distanceTo(
                new THREE.Vector3(goal.location.x, agent.object.position.y, goal.location.z)
            );
            
            if (distance < 2) {
                agent.goals.shift(); // Remove completed goal
                agent.currentAction = null;
            }
        }
    }

    /**
     * Update agent state
     */
    updateState(agent) {
        agent.state.position = agent.object.position.clone();
        
        // Energy regeneration
        if (agent.currentAction?.type === 'rest') {
            agent.state.energy = Math.min(100, agent.state.energy + 0.5);
        } else {
            agent.state.energy = Math.max(0, agent.state.energy - 0.05);
        }

        // Mood based on state
        if (agent.state.health < 30 || agent.state.energy < 20) {
            agent.state.mood = 'stressed';
        } else if (agent.perception.threats.length > 0) {
            agent.state.mood = 'alert';
        } else if (agent.perception.opportunities.length > 0) {
            agent.state.mood = 'curious';
        } else {
            agent.state.mood = 'neutral';
        }
    }

    /**
     * Update agent memory
     */
    updateMemory(agent) {
        const memoryEntry = {
            time: Date.now(),
            position: agent.object.position.clone(),
            action: agent.currentAction?.type,
            nearbyAgents: agent.perception.nearbyAgents.length,
            state: { ...agent.state }
        };

        agent.memory.push(memoryEntry);

        // Limit memory size
        if (agent.memory.length > this.config.memorySize) {
            agent.memory.shift();
        }
    }

    /**
     * Helper: Check if another agent is a threat
     */
    isThreat(agent, otherAgent) {
        if (otherAgent.type === 'enemy') return true;
        if (otherAgent.personality === 'aggressive') return true;
        if (otherAgent.state.mood === 'aggressive') return true;
        return false;
    }

    /**
     * Helper: Check if object is an opportunity
     */
    isOpportunity(agent, obj) {
        if (obj.type === 'resource' && agent.state.energy < 50) return true;
        if (obj.type === 'goal' && agent.goals.some(g => g.type === obj.type)) return true;
        return false;
    }

    /**
     * Helper: Find safe location away from threats
     */
    findSafeLocation(agent) {
        const currentPos = agent.object.position;
        const threats = agent.perception.nearbyAgents.filter(a => 
            agent.perception.threats.includes(a.id)
        );

        if (threats.length === 0) {
            return { x: currentPos.x, z: currentPos.z };
        }

        // Calculate direction away from threats
        let avoidX = 0, avoidZ = 0;
        for (const threat of threats) {
            const dx = currentPos.x - threat.position.x;
            const dz = currentPos.z - threat.position.z;
            avoidX += dx;
            avoidZ += dz;
        }

        return {
            x: currentPos.x + avoidX * 2,
            z: currentPos.z + avoidZ * 2
        };
    }

    /**
     * Helper: Calculate flee target
     */
    calculateFleeTarget(agent, threat) {
        const dx = agent.object.position.x - threat.position.x;
        const dz = agent.object.position.z - threat.position.z;
        const distance = Math.sqrt(dx * dx + dz * dz);

        return {
            x: agent.object.position.x + (dx / distance) * 20,
            z: agent.object.position.z + (dz / distance) * 20
        };
    }

    /**
     * Add goal to agent
     */
    addGoal(agentId, goal) {
        const agent = this.agents.get(agentId);
        if (agent) {
            agent.goals.push(goal);
            return true;
        }
        return false;
    }

    /**
     * Get agent data
     */
    getAgent(agentId) {
        return this.agents.get(agentId);
    }

    /**
     * Get all agents
     */
    getAllAgents() {
        return Array.from(this.agents.values());
    }

    /**
     * Export agent data for analysis
     */
    exportAgentData(agentId) {
        const agent = this.agents.get(agentId);
        if (!agent) return null;

        return {
            id: agent.id,
            type: agent.type,
            personality: agent.personality,
            state: { ...agent.state },
            memory: [...agent.memory],
            decisionHistory: [...agent.decisionHistory],
            goals: [...agent.goals]
        };
    }
}
