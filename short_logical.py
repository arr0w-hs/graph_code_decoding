import numpy as np
from stabiliser_code import update_tableau,append_rows_to_tableau,\
find_logical_op_basis,append_logical_to_tableau,tableau_list_to_matrix,rank_F2,construct_Omega_Matrix,\
compute_Pauli_weight,y_positions
from ldpc import BpDecoder, bposd_decoder
from galois import GF2
import networkx as nx
from gspf_ilp import create_graph_code


def pass_to_decoder(T,X_logicals,CSS:bool,logical_qubit:int=0,rounds:int=1,max_iter:int=100):#T with logical X appended. So far only works for CSS codes
    #assumes T is already in its reduced form! Full rank!
    """logical_qubit is 0-indexed"""
    #finds a short X operator 
    if rank_F2(T)<T.shape[0]:

        raise ValueError('T not full rank')
    
    n_qubits=T.shape[1]//2 #integer division

    num_gen=T.shape[0]

    num_gen_x=num_gen//2

    #X_logicals,Z_logicals,logicals=find_logical_op_basis(T,n_qubits)

    #if X_logicals==None or Z_logicals==None:
        #raise TypeError('X_logicals or Z_logicals are None. Means that T somewhere turned into a non-CSS code.')
    
    if isinstance(X_logicals,list):
        X_logicals=tableau_list_to_matrix(X_logicals)

    if CSS:
        T=T[:num_gen_x,:n_qubits]
        num_gen=num_gen_x

 
    H=append_rows_to_tableau(T,X_logicals)
    
    syndrome = np.zeros(H.shape[0], dtype=int)
    syndrome[num_gen + logical_qubit] = 1  # 1 at the logical qubit's row, after the stabilizer rows

    syndrome_1d = syndrome.flatten()

    syndrome_1d = syndrome_1d.astype(np.uint8)
    

    if CSS:
        decoder = BpDecoder(H, max_iter=max_iter)
        result = decoder.decode(syndrome_1d)

    else:
        Omega=construct_Omega_Matrix(n_qubits)
        H_eff= (H.astype(int) @ Omega.astype(int)) % 2
        H_eff = H_eff.astype(np.uint8)
        channel_probs = np.full(2 * n_qubits, 0.1)

        best = None
        best_weight = np.inf

        for _ in range(rounds):

            decoder = bposd_decoder(H_eff, channel_probs=channel_probs,
                                max_iter=max_iter, bp_method="ms",
                                osd_method="osd_cs", osd_order=6)
            result = decoder.decode(syndrome_1d)
            assert np.array_equal((H_eff @ result) % 2, syndrome_1d), \
    "decoded result does not reproduce the syndrome"
assert result.any(), \
    "decoder returned all-zero (no operator found)"
            pauli_weight=compute_Pauli_weight(result)
 
            if pauli_weight < best_weight:
                best_weight = pauli_weight
                best = result.copy()
            
            y_mask = y_positions(result)
            channel_probs[:n_qubits][y_mask] = np.minimum(channel_probs[:n_qubits][y_mask] * 2, 0.49) 
            channel_probs[n_qubits:][y_mask] = np.minimum(channel_probs[n_qubits:][y_mask] * 2, 0.49)
 
        result=best

    return result

 
def main():

    numq = 30
    # g = nx.erdos_renyi_graph(numq, 0.7)
    g = nx.cycle_graph(numq)
    g = nx.to_numpy_array(g, dtype = np.uint16)

    xlogi, zlogi, stabi = create_graph_code(g)
    T=some_code #TODO
    n_qubits=2
    X_logicals,Z_logicals,logicals=find_logical_op_basis(T,n_qubits)

    short_z=pass_to_decoder(T,X_logicals)
    return short_z



if __name__ == "__main__":
    numq = 10
    # g = nx.erdos_renyi_graph(numq, 0.7)
    g = nx.cycle_graph(numq)
    g = nx.to_numpy_array(g, dtype = np.uint16)

    xlogi, zlogi, stabi = create_graph_code(g)

    r= rank_F2(stabi)


    print(r)

    
 
    