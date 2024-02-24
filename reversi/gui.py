# coding: utf-8
import numpy as np
import flet as ft
from reversi import Reversi
import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
from mcts import MCTS


def create_disc_content(player):
    if player == 0:
        return None
    elif player == 1:
        return ft.Container(width=50, height=50, border_radius=50, bgcolor=ft.colors.BLACK)
    elif player == -1:
        return ft.Container(width=50, height=50, border_radius=50, bgcolor=ft.colors.WHITE)


def get_done_message(reversi, state):
    num_b, num_w = reversi.count_discs(state)
    if num_b == num_w:
        return f"DRAW (B:{num_b}, W:{num_w})"
    elif num_b > num_w:
        return f"BLACK WINS (B:{num_b}, W:{num_w})"
    elif num_b < num_w:
        return f"WHITE WINS (B:{num_b}, W:{num_w})"


def main(page: ft.Page):
    page.title = "MCTS REVERSI"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.theme_mode = ft.ThemeMode.DARK
    page.fonts = {"MPlus1cLight": "./fonts/M_PLUS_Rounded_1c/MPLUSRounded1c-Light.ttf"}

    def state_2d_to_board(state_2d):
        """
        2次元ndarray(-1,0,1)から盤面を作って返す
        """
        board = []
        for row in range(state_2d.shape[0]):
            board_row = []
            for col in range(state_2d.shape[1]):
                board_row.append(
                    ft.Container(
                        key=f"{row},{col}",
                        width=50,
                        height=50,
                        bgcolor=ft.colors.GREEN,
                        on_click=cnt_clicked,
                        content=create_disc_content(state_2d[row, col]),
                    )
                )
            board.append(board_row)
        return board

    def e_ai(mcts, state, current_player, reversi):
        """
        エーアイで次の手を選択する
        """
        policy = mcts.search(state, current_player, num_simulations=50, game=reversi)
        action = np.random.choice(np.where(np.array(policy) == max(policy))[0])
        return action

    def update_board(board, reversi, state, idx, next_state):
        """
        盤面の表示を更新する
        """
        if idx != reversi.action_noop:
            r, c = reversi._convert_index_1d_to_2d(idx)
        else:
            r, c = -1, -1

        # 背景リセット (前の着手マスの色を戻す)
        for board_row in board:
            for container in board_row:
                container.bgcolor = ft.colors.GREEN

        # 着手マスの色変更とディスクひっくり返し
        for changed_idx in np.where(state != next_state)[0]:
            changed_row, changed_col = reversi._convert_index_1d_to_2d(changed_idx)
            board[changed_row][changed_col].content = create_disc_content(next_state[changed_idx])
            if (changed_row == r) and (changed_col == c):
                board[changed_row][changed_col].bgcolor = ft.colors.AMBER

    def cnt_clicked(e):
        """
        マス目をクリックしたときのイベント
        """
        board = page.session.get("board")
        human = page.session.get("human")
        reversi = page.session.get("reversi")
        state = page.session.get("state")
        current_player = page.session.get("current_player")
        mcts = page.session.get("mcts") if page.session.contains_key("mcts") else None
        r_, c_ = [int(t) for t in e.control.key.split(",")]

        # 合法手判定
        idx_ = reversi._convert_index_2d_to_1d(row=r_, col=c_)
        is_legal_move = reversi.is_legal_move(state, idx_, current_player)

        if is_legal_move:
            # 合法なら手を進める
            next_state, is_done = reversi.step(state, idx_, current_player)
            update_board(board, reversi, state, idx_, next_state)

            state = next_state
            current_player = -current_player
            cnt_next.content = create_disc_content(current_player)
            txt_next.value = ": YOU" if human == current_player else ": CPU"

            if (is_done == False) and (mcts is not None):
                # エーアイの手番
                page.update()  # 人間の手を表示に反映する

                e_ai_idx = e_ai(mcts, state, current_player, reversi)
                next_state, is_done = reversi.step(state, e_ai_idx, current_player)
                update_board(board, reversi, state, e_ai_idx, next_state)

                state = next_state
                current_player = -current_player
                cnt_next.content = create_disc_content(current_player)
                txt_next.value = ": YOU" if human == current_player else ": CPU"
                page.session.set("mcts", mcts)

            # 次の手がパスしかない場合はボタンを表示
            legal_moves = reversi.get_legal_moves(state, current_player)
            if (is_done == False) and ([reversi.action_noop] == legal_moves):
                btn_pass.disabled = False
            # 終了した場合は表示
            if is_done:
                txt_done.visible = True
                txt_done.value = get_done_message(reversi, state)

            page.session.set("board", board)
            page.session.set("current_player", current_player)
            page.session.set("state", state)
            page.session.set("is_done", is_done)
            page.update()
        else:
            page.dialog = ilegal_dialog
            ilegal_dialog.open = True
            page.update()

    def btn_pass_clicked(e):
        """
        パスボタン押下イベ
        """
        board = page.session.get("board")
        human = page.session.get("human")
        reversi = page.session.get("reversi")
        state = page.session.get("state")
        current_player = page.session.get("current_player")
        mcts = page.session.get("mcts") if page.session.contains_key("mcts") else None

        next_state, is_done = reversi.step(state, reversi.action_noop, current_player)
        update_board(board, reversi, state, reversi.action_noop, next_state)

        state = next_state
        current_player = -current_player
        cnt_next.content = create_disc_content(current_player)
        txt_next.value = ": YOU" if human == current_player else ": CPU"

        if (is_done == False) and (mcts is not None):
            # エーアイの手番
            page.update()  # 先に表示を更新しておく

            e_ai_idx = e_ai(mcts, state, current_player, reversi)
            next_state, is_done = reversi.step(state, e_ai_idx, current_player)
            update_board(board, reversi, state, reversi.action_noop, next_state)

            state = next_state
            current_player = -current_player
            cnt_next.content = create_disc_content(current_player)
            txt_next.value = ": YOU" if human == current_player else ": CPU"
            page.session.set("mcts", mcts)

        #
        legal_moves = reversi.get_legal_moves(state, current_player)
        if [reversi.action_noop] != legal_moves:
            btn_pass.disabled = True
        if is_done:
            txt_done.visible = True
            txt_done.value = get_done_message(reversi, state)

        page.session.set("board", board)
        page.session.set("current_player", current_player)
        page.session.set("state", state)
        page.session.set("is_done", is_done)
        page.update()

    def btn_start_clicked(e):
        """
        スタートボタン押下: モデル読み込みや盤面初期化・表示
        """
        # restart時: 既にあるボードを消す
        num_board_rows = np.sum(
            [
                1 if (control.key is not None) and (control.key.startswith("board_row")) else 0
                for control in page.controls
            ]
        )
        for _ in range(num_board_rows):
            page.controls.pop()
        if num_board_rows > 0:
            page.update()

        # ここからstart
        num_rows, num_cols = [int(k) for k in dd.value.split(",")]
        human = 1 if dd_teban.value == "b" else -1

        # 盤面初期化
        reversi = Reversi(num_rows, num_cols)
        state = reversi.initialize()
        current_player = 1
        # モデル等の初期化処理
        model = keras.models.load_model(f"model_{num_rows}x{num_cols}.keras")
        mcts = MCTS(model)

        # ボードの表示
        board = state_2d_to_board(reversi.get_state_2d(state))
        for r_, board_row in enumerate(board):
            page.add(ft.Row(board_row, key=f"board_row_{r_}", alignment=ft.MainAxisAlignment.CENTER))

        btn_pass.visible = True
        btn_pass.disabled = True
        cnt_next.visible = True
        cnt_next.content = create_disc_content(1)
        txt_next.visible = True
        txt_next.value = ": YOU" if human == current_player else ": CPU"
        txt_done.visible = False
        txt_done.value = ""

        if human == -1:
            page.update()  # 先に表示を更新しておく

            e_ai_idx = e_ai(mcts, state, current_player, reversi)
            next_state, _ = reversi.step(state, e_ai_idx, current_player)
            update_board(board, reversi, state, e_ai_idx, next_state)

            state = next_state
            current_player = -current_player
            cnt_next.content = create_disc_content(current_player)
            txt_next.value = ": YOU" if human == current_player else ": CPU"
            page.session.set("mcts", mcts)

        page.session.set("human", human)
        page.session.set("reversi", reversi)
        page.session.set("state", state)
        page.session.set("is_done", False)
        page.session.set("current_player", current_player)

        page.session.set("board", board)
        page.session.set("mcts", mcts)
        e.control.text = "Restart"
        page.update()

    ilegal_dialog = ft.AlertDialog(
        modal=False, title=ft.Text("無効な手です", text_align=ft.TextAlign.CENTER, font_family="MPlus1cLight")
    )

    dd = ft.Dropdown(
        value="6,6",
        width=200,
        options=[
            ft.dropdown.Option("6,6", text="6 x 6"),
        ],
    )
    dd_teban = ft.Dropdown(
        value="b",
        width=200,
        options=[
            ft.dropdown.Option("b", text="BLACK"),
            ft.dropdown.Option("w", text="WHITE"),
        ],
    )
    btn_start = ft.ElevatedButton(text="Start", on_click=btn_start_clicked)
    btn_pass = ft.FilledButton(text="Pass", on_click=btn_pass_clicked, visible=False, disabled=True)
    txt_done = ft.Text("", visible=False, font_family="MPlus1cLight")
    cnt_next = ft.Container(width=25, height=25, bgcolor=ft.colors.GREEN, visible=False)
    txt_next = ft.Text("", visible=False, font_family="MPlus1cLight")

    page.add(
        ft.Row(
            [
                dd,
                dd_teban,
                btn_start,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        )
    )
    page.add(
        ft.Row(
            [
                cnt_next,
                txt_next,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        )
    )
    page.add(ft.Row([btn_pass], alignment=ft.MainAxisAlignment.CENTER))
    page.add(ft.Row([txt_done], alignment=ft.MainAxisAlignment.CENTER))


ft.app(target=main)
