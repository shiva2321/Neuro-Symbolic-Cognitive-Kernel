package nsck;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.animation.AnimationTimer;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;

public class SnakeCanvas extends Pane {
    
    private Canvas canvas;
    private ZContext context;
    private ZMQ.Socket videoSocket; // Sends state/video
    private ZMQ.Socket actionSocket; // Receives action
    
    private int[] snakePos = {5, 5};
    private int[] applePos = {2, 2};
    
    public SnakeCanvas() {
        canvas = new Canvas(400, 400);
        this.getChildren().add(canvas);
        draw();
    }
    
    public void start() {
        Thread netThread = new Thread(() -> {
            try (ZContext ctx = new ZContext()) {
                // DEALER/ROUTER or REQ/REP?
                // Python is REP on 5555
                ZMQ.Socket sender = ctx.createSocket(org.zeromq.SocketType.REQ);
                sender.connect("tcp://localhost:5555");
                
                // Python is PUB on 5556
                ZMQ.Socket subscriber = ctx.createSocket(org.zeromq.SocketType.SUB);
                subscriber.connect("tcp://localhost:5556");
                subscriber.subscribe(ZMQ.SUBSCRIPTION_ALL);
                
                while (!Thread.currentThread().isInterrupted()) {
                    // 1. Send Frame/State
                    sender.send("GRID_DATA_JSON");
                    
                    // 2. Wait for ACK (synchronization)
                    String ack = sender.recvStr(0);
                    
                    // 3. Check for Action Update (Non-blocking usually, but let's block for step)
                    // If Python PUBs immediately after REP, we might catch it.
                    // But PUB/SUB is async. 
                    // Better design: The REP should contain the ACTION if purely 1:1.
                    // But adhering to spec (5556 Action Out):
                    
                    String action = subscriber.recvStr(ZMQ.DONTWAIT);
                    if (action != null) {
                        updateState(action);
                    }
                    
                    try { Thread.sleep(100); } catch (InterruptedException e) { break; }
                }
            }
        });
        netThread.setDaemon(true);
        netThread.start();
        
        // Animation Loop for smoothing if needed
        new AnimationTimer() {
            public void handle(long now) {
                draw();
            }
        }.start();
    }
    
    private void updateState(String action) {
        // Simple logic to move snake
        // UP, DOWN, LEFT, RIGHT
        switch(action) {
            case "UP": snakePos[1]--; break;
            case "DOWN": snakePos[1]++; break;
            case "LEFT": snakePos[0]--; break;
            case "RIGHT": snakePos[0]++; break;
        }
        // Wrap around
        if (snakePos[0] < 0) snakePos[0] = 9;
        if (snakePos[0] > 9) snakePos[0] = 0;
        if (snakePos[1] < 0) snakePos[1] = 9;
        if (snakePos[1] > 9) snakePos[1] = 0;
    }
    
    private void draw() {
        GraphicsContext gc = canvas.getGraphicsContext2D();
        gc.setFill(Color.BLACK);
        gc.fillRect(0, 0, 400, 400);
        
        // Draw Grid 10x10 -> 40x40 pixels per cell
        int cellSize = 40;
        
        // Draw Snake
        gc.setFill(Color.GREEN);
        gc.fillRect(snakePos[0]*cellSize, snakePos[1]*cellSize, cellSize, cellSize);
        
        // Draw Apple
        gc.setFill(Color.RED);
        gc.fillRect(applePos[0]*cellSize, applePos[1]*cellSize, cellSize, cellSize);
    }
}
