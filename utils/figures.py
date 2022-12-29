import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class Visualization:

    @staticmethod
    def adjusted_steps(epochs, loss_history):
        for i in range(1, len(loss_history)):
            if abs(loss_history[i] - loss_history[i-1]) < 0.1:
                break
        if i > 50:
            for i in range(1, len(loss_history)):
                if abs(loss_history[i] - loss_history[i-1]) < 0.5:
                    break
        if i > 100 and i <= 50:
            for i in range(1, len(loss_history)):
                if abs(loss_history[i] - loss_history[i-1]) < 1:
                    break
        if i > 150 and i <=100:
            for i in range(1, len(loss_history)):
                if abs(loss_history[i] - loss_history[i-1]) < 2:
                    break
        if i > 200 and i<=150:
            for i in range(1, len(loss_history)):
                if abs(loss_history[i] - loss_history[i-1]) < 3:
                    break

        if len(loss_history) > epochs:
            if epochs - i < 50:
                adjusted_steps = np.r_[np.arange(i+1), epochs]
            else:
                rest_n = 100
                rest_steps = np.arange(i+1, len(loss_history), rest_n)
                adjusted_steps = np.r_[np.arange(i+1), rest_steps, epochs]
        else:
            if len(loss_history) - i < 50:
                adjusted_steps = np.arange(i+1)
            else:
                rest_n = 100
                rest_steps = np.arange(i+1, len(loss_history), rest_n)
                adjusted_steps = np.r_[np.arange(i+1), rest_steps]

        # print("到达第{}步就跳出来".format(i))
        # print("loss history长度:{}".format(len(loss_history)))
        # rest_steps = np.arange(i+1, len(loss_history), rest_n)
        # adjusted_steps = np.r_[np.arange(i+1), rest_steps]
        # print(adjusted_steps)

        return adjusted_steps

    @staticmethod
    def plot_animation_reg(x, y, epochs, weight_history, loss_history, best_weights, 
                    best_error, loss_func=None, penalty=None):
        X = np.c_[np.ones(len(x)), x]
        n = len(weight_history)
        xs = np.linspace(x.min(), x.max(), n)
        xs_b = np.c_[np.ones(n), xs]
        y_target = xs_b @ best_weights
        ys = []
        for w in weight_history:
            ys.append(xs_b @ w)

        # 系数
        W = weight_history[-1]

        max_interval = np.abs(np.max(weight_history) - np.min(weight_history))
        w0n_min = W[0] - max_interval
        w0n_max = W[0] + max_interval
        w1n_min = W[1] - max_interval
        w1n_max = W[1] + max_interval
        n = 80
        w0n = 1.01 * np.linspace(w0n_min, w0n_max, n)
        w1n = 1.01 * np.linspace(w1n_min, w1n_max, n)
        ww0n, ww1n = np.meshgrid(w0n, w1n)
        Wn = np.c_[ww0n.ravel(), ww1n.ravel()]

        lossn = np.array([loss_func(W) for W in Wn]).reshape(ww0n.shape)
        w0_path = np.array([w[0] for w in weight_history])
        w1_path = np.array([w[1] for w in weight_history])

        scatter_trace = go.Scatter(
            x=x.ravel(),
            y=y,
            mode='markers',
            marker=dict(color='#2221f6')
        )

        line_trace = go.Scatter(
            x=xs,
            y=y_target,
            mode='lines',
            line_color='#ff1414'
        )
        best_error_trace = go.Scatter(
            x=np.arange(len(loss_history)),
            y=[best_error] * len(loss_history),
            mode='lines',
            line=dict(color='#ff1414', dash='dash')
        )

        params3d_trace = go.Surface(
            x=w0n,
            y=w1n,
            z=lossn,
            showscale=False,
            opacity=0.95, 
            colorscale='bluered'
        )

        best_param3d_trace = go.Scatter3d(
            x=[best_weights[0]],
            y=[best_weights[1]],
            z=[best_error],
            mode='markers',
            marker=dict(color='#ff1414', size=15, symbol='cross')
        )

        # contour_trace = go.Contour(
        #     x=w0n, 
        #     y=w1n, 
        #     z=lossn,
        #     colorscale='rainbow',
        #     showscale=False
        # )

        contour_trace = go.Scatter(
            x=ww0n.ravel(), 
            y=ww1n.ravel(), 
            mode='markers',
            marker=dict(size=10, color=lossn.ravel(), 
                        symbol='square',
                        colorscale='bluered'
                        )
        )

        # path2d_trace = go.Scatter(
        #     x=w0_path,
        #     y=w1_path,
        #     mode="lines",
        #     line=dict(color="#47ff47", width=4),
        # )


        best_param2d_trace = go.Scatter(
            x=[best_weights[0]],
            y=[best_weights[1]],
            marker=dict(color="#ff1414", size=15, symbol='cross'),
        )
        # fig = make_subplots(
        #         rows=2, cols=4, 
        #         specs=[[{"type": "scatter",'colspan': 2}, None, 
        #                 {"type": "scatter",'colspan': 2}, None],
        #             [None, {"type": "surface",'colspan': 2}, None, None]],
        #         horizontal_spacing=0.1,
        #         vertical_spacing=0.03,
        #         # column_widths=[0.0, 0.45, 0.35, 0.0],
        #         column_widths=[0.0, 0.45, 0.4, 0.0],
        #         row_heights=[0.4, 0.6]
        # )

        fig = make_subplots(
                rows=2, cols=2, 
                specs=[[{"type": "scatter",'colspan': 1}, {"type": "scatter",'colspan': 1}],
                    [{"type": "surface",'colspan': 1},{"type": "scatter",'colspan': 1}]],
                horizontal_spacing=0.1,
                vertical_spacing=0.08,
                column_widths=[0.5, 0.5],
                row_heights=[0.4, 0.6]
        )
        height = 900
        if penalty is not None:
            fig = make_subplots(
                rows=3, cols=2, 
                specs=[[{"type": "scatter",'colspan': 1}, {"type": "scatter",'colspan': 1}],
                    [{"type": "surface",'colspan': 1},{"type": "scatter",'colspan': 1}],
                    [{"type": "contour",'colspan': 1}, None]],
                horizontal_spacing=0.1,
                vertical_spacing=0.06,
                column_widths=[0.5, 0.5],
                row_heights=[0.4, 0.6, 0.6])
            height = 1400
            msen = np.array([penalty[0](W) for W in Wn]).reshape(ww0n.shape)
            mse_contour_trace = go.Contour(
                x=w0n, 
                y=w1n, 
                z=msen,
                showscale=False,
                colorscale='greens'
            )
            penaltyn = np.array([penalty[1](W) for W in Wn]).reshape(ww0n.shape)
            # contraint_bound = np.sum(np.abs(weight_history[-1]))
            penalty_trace = go.Contour(
                x=w0n, 
                y=w1n, 
                z=penaltyn,
                opacity=0.6,
                showscale=False,
                contours=dict(),
                ncontours=150,
                colorscale='reds',              
                name="L1范数"
            )
            fig.add_trace(mse_contour_trace, row=3, col=1)
            fig.add_trace(penalty_trace, row=3, col=1)
            fig.add_trace(path2d_trace, row=3, col=1)
            fig.add_trace(best_trace, row=3, col=1)
            fig.update_layout(
                xaxis4=dict(title="w1", showline=True, gridcolor='gray'),
                yaxis4=dict(title="w2", showline=True, gridcolor='gray')
            )
        
        fig.add_trace(line_trace, row=1, col=1)
        fig.add_trace(scatter_trace, row=1, col=1)
        fig.add_trace(line_trace, row=1, col=1)

        fig.add_trace(best_error_trace, row=1, col=2)
        fig.add_trace(best_error_trace, row=1, col=2)
        fig.add_trace(params3d_trace, row=2, col=1)
        fig.add_trace(best_param3d_trace, row=2, col=1)
        fig.add_trace(best_param3d_trace, row=2, col=1)
        # fig.add_trace(path_trace, row=2, col=1)
        fig.add_trace(contour_trace, row=2, col=2)
        fig.add_trace(contour_trace, row=2, col=2)
        # fig.add_trace(path2d_trace, row=2, col=2)
        # fig.add_trace(best_trace, row=2, col=2)
        # fig.add_trace(best_trace, row=2, col=2)
        fig.add_trace(best_param2d_trace, row=2, col=2)
        fig.add_trace(best_param2d_trace, row=2, col=2)
        
        
        fig.update_layout(
            xaxis1=dict(title="x", showline=True, gridcolor='gray'),
            yaxis1=dict(title="y", showline=True, gridcolor='gray'),
            xaxis2=dict(title="步数", range=[0, len(loss_history)], 
                        showline=True, gridcolor='gray', zeroline=False),
            yaxis2=dict(title="误差", range=[best_error-50, np.max(loss_history)+10], 
                        showline=True, gridcolor='gray', zeroline=False),
            xaxis3=dict(title="w1", showline=False, showgrid=False, zeroline=False),
            yaxis3=dict(title="w2", showline=False, showgrid=False, zeroline=False),
            font_color="white",
        )
        fig.update_scenes(
            xaxis_visible=False, 
            yaxis_visible=False,
            zaxis_visible=False,
            camera=dict(eye=dict(x=0.5, y=1.1, z=1.0)))

        adjusted_steps = Visualization.adjusted_steps(epochs, loss_history)
        frames = []
        for k in adjusted_steps.tolist():
            frames.append(dict(
                name=k,
                data=[
                    go.Scatter(visible=True),   
                    go.Scatter(visible=True),   
                    go.Scatter(
                        x=xs,
                        y=np.sort(ys[k]),
                        mode='lines',
                        line_color='#47ff47'
                    ),
                    go.Scatter(visible=True), 
                    go.Scatter(
                        x=np.arange(len(loss_history))[:k],
                        y=loss_history[:k],
                        mode='lines',
                        line=dict(color='#47ff47', dash='solid')
                    ), 
                    go.Surface(visible=True),
                    go.Scatter3d(visible=True),
                    go.Scatter3d(
                        x=w0_path[:k],
                        y=w1_path[:k],
                        z=np.array(loss_history[:k]) + 1,
                        mode="lines",
                        line=dict(color="#47ff47", width=8),
                        # marker=dict(color="#47ff47", width=8, symbol='circle')
                    ),
                    # go.Scatter3d(visible=True),
                    # go.Scatter3d(
                    #     x=[w0_path[k]],
                    #     y=[w1_path[k]],
                    #     z=np.array([loss_history[k]]) + 100,
                    #     mode="markers",
                    #     marker=dict(color="#f5f621", size=8, symbol='circle')
                    # ),
                    # go.Scatter(visible=True),
                    go.Scatter(visible=True),
                    go.Scatter(
                        x=w0_path[:k],
                        y=w1_path[:k],
                        mode='lines',
                        line=dict(color="#47ff47", width=4)
                    ),
                    
                    # go.Scatter(
                    #     x=[w0_path[k]],
                    #     y=[w1_path[k]],
                    #     mode="markers",
                    #     marker=dict(color="#f5f621", size=10, symbol='circle')
                    # ),
                ],
                traces=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12],
                layout=go.Layout(
                    title=dict(
                        text="步数：{}      直线公式：y={}x+{}      误差：{}".format(k, 
                            np.round(w0_path[k], 2), np.round(w1_path[k], 2), 
                            np.round(loss_history[k], 2)), 
                        font=dict(family="Courier New, monospace", color="white"),
                        x=0.5,
                        y=0.97),
                    ), 
                )
            )
        fig.update(frames=frames)


        # 添加目标测量的文本标注
        fig.add_annotation(
            x=0,
            y=0,
            xref="x1",
            yref="y1",
            text="目标直线公式：y={}x+{}".format(np.round(best_weights[0], 2),
                    np.round(best_weights[1]), 2),
            showarrow=True,
            font=dict(
                family="Courier New, monospace",
                size=14,
                color="#ffffff"
                ),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=-110,
            ay=-40,
        )
        
        fig.add_annotation(
            x=best_weights[0],
            y=best_weights[1],
            xref="x3",
            yref="y3",
            text="目标权重：w1={},w2={}".format(np.round(best_weights[0], 2),
                    np.round(best_weights[1]), 2),
            showarrow=True,
            font=dict(
                family="Courier New, monospace",
                size=14,
                color="#ffffff"
                ),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=20,
            ay=-30,
        )

        fig.add_annotation(
            x= int(epochs/10),
            y=best_error,
            xref="x2",
            yref="y2",
            text="目标误差：{}".format(np.round(best_error, 2)),
            showarrow=True,
            font=dict(
                family="Courier New, monospace",
                size=14,
                color="#ffffff"
                ),
            align="right",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=17,
            ay=-28,
        )

        fig.update_layout(
            width=1200,
            height=height,
            margin=dict(l=100, r=50, t=15, b=20),
            showlegend=False,
            updatemenus=[dict(
                    type="buttons", 
                    buttons=[dict(label="现场训练",
                            method="animate",
                            args=[None, 
                                {"frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0}}])],
                    x=0.1,
                    y=1.06,
                    bgcolor='black',
                    bordercolor='black',
                    font=dict(color='#E95420', family="sans serif", size=14))
                ],
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            
        )

        return fig

    @staticmethod
    def plot_animation_1dcls(x, y, epochs, weight_history, loss_history, best_weights, 
                    best_error, loss_func=None, penalty=None):
        X = np.c_[np.ones(len(x)), x]
        n = len(weight_history)
        xs = np.linspace(x.min(), x.max(), n)
        xs_b = np.c_[np.ones(n), xs]
        y_target = 1 / (1 + np.exp(-xs_b @ best_weights))
        ys = []
        for w in weight_history:
            ys.append(1 / (1 + np.exp(-xs_b @ w)))

        # 系数
        W = weight_history[-1]

        max_interval = np.abs(np.max(weight_history) - np.min(weight_history))
        w0n_min = W[0] - max_interval
        w0n_max = W[0] + max_interval
        w1n_min = W[1] - max_interval
        w1n_max = W[1] + max_interval
        n = 80
        w0n = 1.01 * np.linspace(w0n_min, w0n_max, n)
        w1n = 1.01 * np.linspace(w1n_min, w1n_max, n)
        ww0n, ww1n = np.meshgrid(w0n, w1n)
        Wn = np.c_[ww0n.ravel(), ww1n.ravel()]

        lossn = np.array([loss_func(W) for W in Wn]).reshape(ww0n.shape)
        w0_path = np.array([w[0] for w in weight_history])
        w1_path = np.array([w[1] for w in weight_history])

        scatter_trace = go.Scatter(
            x=x.ravel(),
            y=y,
            mode='markers',
            marker=dict(color='#2221f6')
        )

        line_trace = go.Scatter(
            x=xs,
            y=y_target,
            mode='lines',
            line_color='#ff1414'
        )

        invert_line_trace = go.Scatter(
            x=xs,
            y=1 - y_target,
            mode='lines',
            line_color='#ff1414'
        )
        best_error_trace = go.Scatter(
            x=np.arange(len(loss_history)),
            y=[best_error] * len(loss_history),
            mode='lines',
            line=dict(color='#ff1414', dash='dash')
        )

        params3d_trace = go.Surface(
            x=w0n,
            y=w1n,
            z=lossn,
            showscale=False,
            opacity=0.95, 
            colorscale='bluered'
        )

        best_param3d_trace = go.Scatter3d(
            x=[best_weights[0]],
            y=[best_weights[1]],
            z=[best_error],
            mode='markers',
            marker=dict(color='#ff1414', size=15, symbol='cross')
        )

        contour_trace = go.Scatter(
            x=ww0n.ravel(), 
            y=ww1n.ravel(), 
            mode='markers',
            marker=dict(size=10, color=lossn.ravel(), 
                        symbol='square',
                        colorscale='bluered'
                        )
        )

        best_param2d_trace = go.Scatter(
            x=[best_weights[0]],
            y=[best_weights[1]],
            marker=dict(color="#ff1414", size=15, symbol='cross'),
        )

        fig = make_subplots(
                rows=2, cols=2, 
                specs=[[{"type": "scatter",'colspan': 1}, {"type": "scatter",'colspan': 1}],
                    [{"type": "surface",'colspan': 1},{"type": "scatter",'colspan': 1}]],
                horizontal_spacing=0.1,
                vertical_spacing=0.08,
                column_widths=[0.5, 0.5],
                row_heights=[0.4, 0.6]
        )
        height = 900
        if penalty is not None:
            fig = make_subplots(
                rows=3, cols=2, 
                specs=[[{"type": "scatter",'colspan': 1}, {"type": "scatter",'colspan': 1}],
                    [{"type": "surface",'colspan': 1},{"type": "scatter",'colspan': 1}],
                    [{"type": "contour",'colspan': 1}, None]],
                horizontal_spacing=0.1,
                vertical_spacing=0.06,
                column_widths=[0.5, 0.5],
                row_heights=[0.4, 0.6, 0.6])
            height = 1400
            msen = np.array([penalty[0](W) for W in Wn]).reshape(ww0n.shape)
            mse_contour_trace = go.Contour(
                x=w0n, 
                y=w1n, 
                z=msen,
                showscale=False,
                colorscale='greens'
            )
            penaltyn = np.array([penalty[1](W) for W in Wn]).reshape(ww0n.shape)
            # contraint_bound = np.sum(np.abs(weight_history[-1]))
            penalty_trace = go.Contour(
                x=w0n, 
                y=w1n, 
                z=penaltyn,
                opacity=0.6,
                showscale=False,
                contours=dict(),
                ncontours=150,
                colorscale='reds',              
                name="L1范数"
            )
            fig.add_trace(mse_contour_trace, row=3, col=1)
            fig.add_trace(penalty_trace, row=3, col=1)
            fig.add_trace(path2d_trace, row=3, col=1)
            fig.add_trace(best_trace, row=3, col=1)
            fig.update_layout(
                xaxis4=dict(title="w1", showline=True, gridcolor='gray'),
                yaxis4=dict(title="w2", showline=True, gridcolor='gray')
            )
        
        # fig.add_trace(invert_line_trace, row=1, col=1)
        fig.add_trace(line_trace, row=1, col=1)
        fig.add_trace(scatter_trace, row=1, col=1)
        fig.add_trace(line_trace, row=1, col=1)
        # fig.add_trace(invert_line_trace, row=1, col=1)

        fig.add_trace(best_error_trace, row=1, col=2)
        fig.add_trace(best_error_trace, row=1, col=2)
        fig.add_trace(params3d_trace, row=2, col=1)
        fig.add_trace(best_param3d_trace, row=2, col=1)
        fig.add_trace(best_param3d_trace, row=2, col=1)
        fig.add_trace(contour_trace, row=2, col=2)
        fig.add_trace(contour_trace, row=2, col=2)
        fig.add_trace(best_param2d_trace, row=2, col=2)
        fig.add_trace(best_param2d_trace, row=2, col=2)
        
        
        fig.update_layout(
            xaxis1=dict(title="x", showline=True, gridcolor='gray'),
            yaxis1=dict(title="y", showline=True, gridcolor='gray'),
            xaxis2=dict(title="步数", range=[0, len(loss_history)], 
                        showline=True, gridcolor='gray', zeroline=False),
            yaxis2=dict(title="误差", range=[best_error-50, np.max(loss_history)+10], 
                        showline=True, gridcolor='gray', zeroline=False),
            xaxis3=dict(title="w1", showline=False, showgrid=False, zeroline=False),
            yaxis3=dict(title="w2", showline=False, showgrid=False, zeroline=False),
            font_color="white",
        )
        fig.update_scenes(
            xaxis_visible=False, 
            yaxis_visible=False,
            zaxis_visible=False,
            camera=dict(eye=dict(x=0.5, y=1.1, z=1.0)))

        adjusted_steps = Visualization.adjusted_steps(epochs, loss_history)
        frames = []
        for k in adjusted_steps.tolist():
            frames.append(dict(
                name=k,
                data=[
                    # go.Scatter(visible=True),  
                    # go.Scatter(visible=True),  
                    go.Scatter(visible=True),   
                    go.Scatter(visible=True),   
                    go.Scatter(
                        x=xs,
                        y=np.sort(ys[k]),
                        mode='lines',
                        line_color='#47ff47'
                    ),
                    go.Scatter(visible=True), 
                    go.Scatter(
                        x=np.arange(len(loss_history))[:k],
                        y=loss_history[:k],
                        mode='lines',
                        line=dict(color='#47ff47', dash='solid')
                    ), 
                    go.Surface(visible=True),
                    go.Scatter3d(visible=True),
                    go.Scatter3d(
                        x=w0_path[:k],
                        y=w1_path[:k],
                        z=np.array(loss_history[:k]) + 1,
                        mode="lines",
                        line=dict(color="#47ff47", width=8),
                        # marker=dict(color="#47ff47", width=8, symbol='circle')
                    ),
                    # go.Scatter3d(visible=True),
                    # go.Scatter3d(
                    #     x=[w0_path[k]],
                    #     y=[w1_path[k]],
                    #     z=np.array([loss_history[k]]) + 100,
                    #     mode="markers",
                    #     marker=dict(color="#f5f621", size=8, symbol='circle')
                    # ),
                    # go.Scatter(visible=True),
                    go.Scatter(visible=True),
                    go.Scatter(
                        x=w0_path[:k],
                        y=w1_path[:k],
                        mode='lines',
                        line=dict(color="#47ff47", width=4)
                    ),
                    
                    # go.Scatter(
                    #     x=[w0_path[k]],
                    #     y=[w1_path[k]],
                    #     mode="markers",
                    #     marker=dict(color="#f5f621", size=10, symbol='circle')
                    # ),
                ],
                traces=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12],
                layout=go.Layout(
                    title=dict(
                        text="步数：{}      直线公式：y=1 / exp(-({}+x{}))      误差：{}".format(k, 
                            np.round(w0_path[k], 2), np.round(w1_path[k], 2), 
                            np.round(loss_history[k], 2)), 
                        font=dict(family="Courier New, monospace", color="white"),
                        x=0.5,
                        y=0.97),
                    ), 
                )
            )
        fig.update(frames=frames)


        # 添加目标测量的文本标注
        fig.add_annotation(
            x=0,
            y=0,
            xref="x1",
            yref="y1",
            text="目标直线公式：y=1/exp(-({}x+{}))".format(np.round(best_weights[0], 2),
                    np.round(best_weights[1]), 2),
            showarrow=True,
            font=dict(
                family="Courier New, monospace",
                size=14,
                color="#ffffff"
                ),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=-110,
            ay=-40,
        )
        
        fig.add_annotation(
            x=best_weights[0],
            y=best_weights[1],
            xref="x3",
            yref="y3",
            text="目标权重：w1={},w2={}".format(np.round(best_weights[0], 2),
                    np.round(best_weights[1]), 2),
            showarrow=True,
            font=dict(
                family="Courier New, monospace",
                size=14,
                color="#ffffff"
                ),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=20,
            ay=-30,
        )

        fig.add_annotation(
            x= int(epochs/10),
            y=best_error,
            xref="x2",
            yref="y2",
            text="目标误差：{}".format(best_error),
            showarrow=True,
            font=dict(
                family="Courier New, monospace",
                size=14,
                color="#ffffff"
                ),
            align="right",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=17,
            ay=-28,
        )

        fig.update_layout(
            width=1200,
            height=height,
            margin=dict(l=100, r=50, t=15, b=20),
            showlegend=False,
            updatemenus=[dict(
                    type="buttons", 
                    buttons=[dict(label="现场训练",
                            method="animate",
                            args=[None, 
                                {"frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0}}])],
                    x=0.1,
                    y=1.06,
                    bgcolor='black',
                    bordercolor='black',
                    font=dict(color='#E95420', family="sans serif", size=14))
                ],
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            
        )

        return fig

    @staticmethod
    def plot_animation_tree(X, y, classifier, best_sets, ys_tracking):
        n = 100
        x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), n)
        x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), n)
        xx1, xx2 = np.meshgrid(x1, x2)
        Xn = np.c_[xx1.ravel(), xx2.ravel()]
        yn_pred = classifier.predict(Xn)
        
        scatter_trace = go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode='markers',
            marker=dict(color=y, colorscale=[[0, 'blue'], [1, 'red']])
        )
        bound_trace = go.Scatter(
            x=xx1.ravel(),
            y=xx2.ravel(),
            mode='markers',
            marker=dict(size=12, symbol='square', color=yn_pred, 
                        colorscale=[[0, 'green'], [1, 'orange']])
        )
        fig = go.Figure()
        fig.add_trace(bound_trace)
        fig.add_trace(scatter_trace)
        
        # frames = []
        # for k in range(len(best_sets)):
        #        frames.append(
        #            dict(
        #                name=k,
        #                data=[
        #                    go.Scatter(
        #                        x=
        #                    )
        #                ]
        #            )
        #        )
        
        return fig


    def plot_animation_cls(X, y, classifier,):
        n = 50
        x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), n)
        x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), n)
        xx1, xx2 = np.meshgrid(x1, x2)
        Xn = np.c_[xx1.ravel(), xx2.ravel()]
        yn_pred = classifier.predict(Xn)
        
        scatter_trace = go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode='markers',
            marker=dict(color=y, colorscale=[[0, "green"], [1, "yellow"]]),
            line=dict(color='white', width=0.8),
        )
        bound_trace = go.Scatter(
            x=xx1.ravel(),
            y=xx2.ravel(),
            mode='markers',
            marker=dict(size=12, symbol='square', color=yn_pred, 
                        colorscale=[[0, 'blue'], [1, 'red']])
        )
        fig = go.Figure()
        fig.add_trace(bound_trace)
        fig.add_trace(scatter_trace)
        # fig.add_trace(bound_trace)
        fig.add_trace(scatter_trace)

        learners = classifier.estimators
        weights = classifier.weights
        print("学习器的数量{}".format(len(learners)))
        frames = []
        n_estimators = classifier.get_params()["n_estimators"]
        if n_estimators >= 20:
            steps = np.r_[np.arange(1, 5), 
                        np.linspace(5, n_estimators - 6, 11).astype(int),
                        np.arange(n_estimators-4, n_estimators+1)]
        
        for k in steps:
            yy = 0
            for learner, weight in zip(learners[:k], weights[:k]):
                yi_pred = np.array([learner.predict(xx.reshape(1, -1)) 
                                for xx in Xn]).reshape(xx1.shape)
                yy += weight * yi_pred
            yy = np.where(yy > 0, 1, 0)
            print(k)
            frames.append(
                dict(
                    name=str(k),
                    data=[
                        go.Scatter(visible=True),
                        go.Scatter(
                            x=xx1.ravel(),
                            y=xx2.ravel(),
                            mode='markers',
                            marker=dict(size=10, symbol='square', color=yy.ravel(), 
                                    colorscale=[[0, 'blue'], [1, 'red']]),
                            opacity=1.0)],
                    traces=[0, 1, 2],
                    layout=go.Layout(
                        title=dict(
                            text="步数：{}".format(k),
                            font=dict(family="Courier New, monospace", color="white"),)
                        )
                )
            )
        fig.update(frames=frames)

        fig.update_layout(
            width=650,
            height=450,
            updatemenus=[dict(
                    type="buttons", 
                    buttons=[dict(label="现场训练",
                            method="animate",
                            args=[None, 
                                {"frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0}}])],
                    x=0.1,
                    y=1.06,
                    bgcolor='black',
                    bordercolor='black',
                    font=dict(color='#E95420', family="sans serif", size=14))
                ],
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            
        )

        return fig