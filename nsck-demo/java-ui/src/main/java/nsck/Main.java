package nsck;

import javafx.application.Application;
import javafx.stage.Stage;
import javafx.scene.Scene;
import javafx.scene.layout.HBox;

public class Main extends Application {

    @Override
    public void start(Stage primaryStage) {
        
        SnakeCanvas snakeGame = new SnakeCanvas();
        ChatWindow chatWindow = new ChatWindow();
        
        HBox root = new HBox(10, snakeGame, chatWindow);
        Scene scene = new Scene(root, 800, 400);
        
        primaryStage.setTitle("NSCK v1.0 - Neuro-Symbolic Snake");
        primaryStage.setScene(scene);
        primaryStage.show();
        
        // Start game loop mechanism inside components
        snakeGame.start();
        chatWindow.start();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
