class SocietalVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.nodes = []; // {id, x, y, radius, color, level, label}
        this.links = []; // {source, target, strength}
        this.camera = { x: 0, y: 0, zoom: 1 };
        this.selectedId = null;
        
        this.resize();
        window.addEventListener('resize', () => this.resize());
        this._initEvents();
    }

    resize() {
        this.canvas.width = this.canvas.parentElement.clientWidth;
        this.canvas.height = this.canvas.parentElement.clientHeight;
        this.camera.x = this.canvas.width / 2;
        this.camera.y = this.canvas.height / 2;
    }

    setData(hierarchy) {
        // Transform hierarchy into flat nodes and links
        const newNodes = [];
        const newLinks = [];
        
        const centerX = 0;
        const centerY = 0;

        hierarchy.forEach((dom, dIdx) => {
            const angle = (dIdx / hierarchy.length) * Math.PI * 2;
            const dist = 300 + Math.random() * 50;
            const dx = Math.cos(angle) * dist;
            const dy = Math.sin(angle) * dist;

            // Domain Node
            newNodes.push({
                id: dom.id,
                x: dx,
                y: dy,
                vx: 0, vy: 0,
                radius: 15 + dom.health * 10,
                color: '#7c4dff',
                level: dom.level,
                label: dom.name,
                type: 'domain'
            });

            // Neighborhoods within Domain
            dom.neighborhoods.forEach((nh, nIdx) => {
                const nAngle = (nIdx / dom.neighborhoods.length) * Math.PI * 2;
                const nDist = 60 + Math.random() * 20;
                const nx = dx + Math.cos(nAngle) * nDist;
                const ny = dy + Math.sin(nAngle) * nDist;

                newNodes.push({
                    id: nh.id,
                    x: nx,
                    y: ny,
                    vx: 0, vy: 0,
                    radius: 5 + Math.sqrt(nh.members_count),
                    color: '#00e5ff',
                    level: 'Neighborhood',
                    label: nh.name,
                    type: 'neighborhood',
                    parentId: dom.id
                });

                newLinks.push({ source: dom.id, target: nh.id, strength: 0.1 });
            });
        });

        this.nodes = newNodes;
        this.links = newLinks;
    }

    _initEvents() {
        let isDragging = false;
        let lastPos = { x: 0, y: 0 };

        this.canvas.onmousedown = (e) => {
            isDragging = true;
            lastPos = { x: e.clientX, y: e.clientY };
            
            // Check selection
            const rect = this.canvas.getBoundingClientRect();
            const mouseX = (e.clientX - rect.left - this.camera.x) / this.camera.zoom;
            const mouseY = (e.clientY - rect.top - this.camera.y) / this.camera.zoom;

            this.selectedId = null;
            for (const node of this.nodes) {
                const d = Math.sqrt((node.x - mouseX)**2 + (node.y - mouseY)**2);
                if (d < node.radius) {
                    this.selectedId = node.id;
                    if (window.onNodeSelect) window.onNodeSelect(node);
                    break;
                }
            }
        };

        window.onmousemove = (e) => {
            if (isDragging) {
                this.camera.x += e.clientX - lastPos.x;
                this.camera.y += e.clientY - lastPos.y;
                lastPos = { x: e.clientX, y: e.clientY };
            }
        };

        window.onmouseup = () => isDragging = false;
        
        this.canvas.onwheel = (e) => {
            const zoomSpeed = 0.001;
            this.camera.zoom -= e.deltaY * zoomSpeed;
            this.camera.zoom = Math.max(0.1, Math.min(5, this.camera.zoom));
            e.preventDefault();
        };
    }

    update() {
        // Simple Physics (Spring and Repulsion)
        const k = 0.05; // spring
        const r = 2000; // repulsion
        
        for (let i = 0; i < this.nodes.length; i++) {
            const n1 = this.nodes[i];
            
            // Repulsion from others
            for (let j = i + 1; j < this.nodes.length; j++) {
                const n2 = this.nodes[j];
                const dx = n2.x - n1.x;
                const dy = n2.y - n1.y;
                const distSq = dx * dx + dy * dy + 0.1;
                const dist = Math.sqrt(distSq);
                
                if (dist < 300) {
                    const fx = (dx / dist) * (r / distSq);
                    const fy = (dy / dist) * (r / distSq);
                    n1.vx -= fx; n1.vy -= fy;
                    n2.vx += fx; n2.vy += fy;
                }
            }
        }

        // Links
        this.links.forEach(link => {
            const s = this.nodes.find(n => n.id === link.source);
            const t = this.nodes.find(n => n.id === link.target);
            if (!s || !t) return;
            
            const dx = t.x - s.x;
            const dy = t.y - s.y;
            const dist = Math.sqrt(dx*dx + dy*dy);
            const force = (dist - 100) * link.strength;
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;
            s.vx += fx; s.vy += fy;
            t.vx -= fx; t.vy -= fy;
        });

        // Apply Velocity
        this.nodes.forEach(n => {
            n.x += n.vx;
            n.y += n.vy;
            n.vx *= 0.9; // damping
            n.vy *= 0.9;
        });
    }

    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.save();
        this.ctx.translate(this.camera.x, this.camera.y);
        this.ctx.scale(this.camera.zoom, this.camera.zoom);

        // Draw Links
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        this.ctx.lineWidth = 1;
        this.links.forEach(link => {
            const s = this.nodes.find(n => n.id === link.source);
            const t = this.nodes.find(n => n.id === link.target);
            if (s && t) {
                this.ctx.beginPath();
                this.ctx.moveTo(s.x, s.y);
                this.ctx.lineTo(t.x, t.y);
                this.ctx.stroke();
            }
        });

        // Draw Nodes
        this.nodes.forEach(node => {
            const isSelected = node.id === this.selectedId;
            
            // Glow effect
            this.ctx.shadowBlur = isSelected ? 20 : 10;
            this.ctx.shadowColor = node.color;
            
            this.ctx.fillStyle = node.color;
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Label
            if (isSelected || this.camera.zoom > 0.8) {
                this.ctx.shadowBlur = 0;
                this.ctx.fillStyle = 'white';
                this.ctx.font = `${10 / this.camera.zoom}px Inter`;
                this.ctx.textAlign = 'center';
                this.ctx.fillText(node.label, node.x, node.y + node.radius + 15);
            }
        });

        this.ctx.restore();
    }

    animate() {
        this.update();
        this.draw();
        requestAnimationFrame(() => this.animate());
    }
}
