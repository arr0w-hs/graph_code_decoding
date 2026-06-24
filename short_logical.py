import numpy as np
from stabiliser_code import update_tableau,append_rows_to_tableau,\
find_logical_op_basis,append_logical_to_tableau,tableau_list_to_matrix,rank_F2
from ldpc import BpDecoder
from galois import GF2

def pass_to_decoder(T,X_logicals,logical_qubit:int=0,max_iter:int=100):#T with logical X appended. So far only works for CSS codes
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

    H_x=T[:num_gen_x,:n_qubits]

 
    H_x=append_rows_to_tableau(H_x,X_logicals)
    
    syndrome = np.zeros(H_x.shape[0], dtype=int)
    syndrome[num_gen_x + logical_qubit] = 1  # 1 at the logical qubit's row, after the stabilizer rows

    syndrome_1d = syndrome.flatten()
    decoder = BpDecoder(H_x, max_iter=100)
    result = decoder.decode(syndrome_1d)

    return result


def main():
    T=some_code #TODO
    n_qubits=2
    X_logicals,Z_logicals,logicals=find_logical_op_basis(T,n_qubits)

    short_z=pass_to_decoder(T,X_logicals)
    return short_z

if __name__ == "__main__":

    
    main()
    