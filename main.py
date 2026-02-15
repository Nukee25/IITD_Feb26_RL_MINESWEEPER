import ollama
import json

def run_engine(engine_model: str, command: dict) -> dict:
    """Send a command to the engine model and return the updated board state."""
    response = ollama.chat(
        model=engine_model,
        messages=[{"role": "user", "content": json.dumps(command)}]
    )
    # Ollama returns a dict with 'message' containing 'content'
    content = response["message"]["content"]
    # print("board: "+content)
    return json.loads(content)

def run_player(player_model: str, board_state: dict) -> dict:
    """Send the current board state to the player model and return its move."""
    response = ollama.chat(
        model=player_model,
        messages=[{"role": "user", "content": json.dumps(board_state)}]
    )
    content = response["message"]["content"]
    # print("player: "+ content)
    return json.loads(content)

def run_game(engine_model="simulator:latest", player_model="player:latest", turns=5):
    # Initialize new game
    board_state = run_engine(engine_model, {"type": "new_game"})
    print("Initial Board:\n", json.dumps(board_state, indent=2))

    for t in range(turns):
        print(f"\n--- Turn {t+1} ---")

        # Player decides next move
        move = run_player(player_model, board_state)
        print("Player Move:\n", move)

        # Engine processes the move
        board_state = run_engine(engine_model, move)
        print("Updated Board:\n", json.dumps(board_state, indent=2))

        # Check for game over
        if any("M" in row for row in board_state["board"]):
            print("Game Over! Mine revealed.")
            break

if __name__ == "__main__":
    run_game()
