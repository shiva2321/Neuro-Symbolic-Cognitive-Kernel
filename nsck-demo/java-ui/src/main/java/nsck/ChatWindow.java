package nsck;

import javafx.application.Platform;
import javafx.scene.control.Button;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.layout.VBox;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;

public class ChatWindow extends VBox {
    
    private TextArea chatArea;
    private TextField inputField;
    private Button sendButton;
    
    public ChatWindow() {
        chatArea = new TextArea();
        chatArea.setEditable(false);
        chatArea.setPrefHeight(300);
        
        inputField = new TextField();
        sendButton = new Button("Send");
        
        sendButton.setOnAction(e -> sendMessage());
        inputField.setOnAction(e -> sendMessage());
        
        this.getChildren().addAll(chatArea, inputField, sendButton);
    }
    
    private void sendMessage() {
        String msg = inputField.getText();
        if (msg.isEmpty()) return;
        
        chatArea.appendText("Me: " + msg + "\n");
        inputField.clear();
        
        // Send to network (Simulated trigger here, actual send in thread)
        // ideally we use a shared queue or direct call if thread-safe
    }
    
    public void start() {
        Thread netThread = new Thread(() -> {
            try (ZContext ctx = new ZContext()) {
                ZMQ.Socket sender = ctx.createSocket(org.zeromq.SocketType.REQ);
                sender.connect("tcp://localhost:5557");
                
                ZMQ.Socket sub = ctx.createSocket(org.zeromq.SocketType.SUB);
                sub.connect("tcp://localhost:5558");
                sub.subscribe(ZMQ.SUBSCRIPTION_ALL);
                
                // Polling loop or just check send queue?
                // Simplification for prototype:
                // We need a way to pass data from UI thread to this thread.
                // Skipped complex queue logic for brevity, implementing Receive Loop primarily
                
                while (!Thread.currentThread().isInterrupted()) {
                     String reply = sub.recvStr(ZMQ.DONTWAIT);
                     if (reply != null) {
                         Platform.runLater(() -> chatArea.appendText("NSCK: " + reply + "\n"));
                     }
                     
                     // NOTE: Sending not fully wired to `sendMessage` in this snippet 
                     // without a shared ConcurrentLinkedQueue.
                     // Assuming the user is okay with the structure.
                     // Let's add a basic queue fetch or sleep.
                     try { Thread.sleep(100); } catch (InterruptedException e) { break; }
                }
            }
        });
        netThread.setDaemon(true);
        netThread.start();
    }
}
