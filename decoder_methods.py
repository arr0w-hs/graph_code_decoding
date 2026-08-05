import numpy as np
import stabiliser_code as sc
from ldpc import BpOsdDecoder
from galois import GF2
import networkx as nx
from tableau import tableau2paulistring
from stabiliser_code import create_graph_code
import tableau as ta
from typing import Literal


def create_error_probs_from_logical(num_qubits: int, logical, p_base: float = 0.1, p_punish: float = 0.02):
    """Build a three-block (X | Z | Y) channel that punishes every single-qubit
    error anticommuting with `logical` (given in symplectic (x|z) form).

    Accepts `logical` as a list, 1D array, or 2D (1, 2n) array.
    """
    # normalise to a flat symplectic vector of length 2*num_qubits
    logical = np.asarray(ta.to_gf2_tableau(logical)).ravel().astype(int)

    if logical.shape[0] != 2 * num_qubits:
        raise ValueError(
            f"Logical defined for {logical.shape[0] // 2} qubits, but num_qubits is {num_qubits}"
        )

    channel_probs = np.full(3 * num_qubits, p_base, dtype=float)  # (x|z|y) three-block notation

    lx = logical[:num_qubits].astype(bool)
    lz = logical[num_qubits:].astype(bool)

    # block a (index [0:n], pairs with sx) = Z-errors: anticommute with logical iff lx
    # block b (index [n:2n], pairs with sz) = X-errors: anticommute iff lz
    # block c (index [2n:3n])              = Y-errors: anticommute iff lx ^ lz  (Y commutes with Y)
    channel_probs[:num_qubits][lx] = p_punish
    channel_probs[num_qubits:2 * num_qubits][lz] = p_punish
    channel_probs[2 * num_qubits:][lx ^ lz] = p_punish

    return channel_probs

################deprecated########################
def make_given_logical_unlikely(channel_probs:np.ndarray, num_qubits:int, logical, p_punish: float = 1e-4):
    """
    Punish the decoder channel on the TXZY support of a found solution vector logical
    (x|z repr), so BP+OSD avoids returning that same vector again. In place.
    """
    logical = ta.to_gf2_tableau(logical)

    if logical.ndim == 2 and logical.shape[1] // 2 != num_qubits:
        raise ValueError(f"Logical on {logical.shape[1]//2} qubits, but num_qubits is {num_qubits}")
    if logical.ndim == 1 and logical.shape[0] // 2 != num_qubits:
        raise ValueError(f"Logical on {logical.shape[0]//2} qubits, but num_qubits is {num_qubits}")

    assert channel_probs.ravel().shape[0] == 3 * num_qubits, \
        f"channel_probs must have {3*num_qubits} entries"

    logical_txzy = ta.turn_tableau_into_TXZYerror(logical)     
    nonzero = np.asarray(logical_txzy).ravel().astype(bool)

    channel_probs[nonzero] = p_punish

    return channel_probs
################do not use#######################

def bias_channel_toward_qubit(channel_probs:np.ndarray, o:int, error_type:Literal["X","Z","Y"],\
                              n_qubits:int, p_low:float=0.95, p_high:float=0.97) -> np.ndarray:

    """Outputs channel probabilities that make it likely that an error happened on qubit o

    Parameters

    -----------
    channel_probs: np.ndarray
        Current error probabilities in three-block form.
    error_type: {X,Z,Y}
        The error type whose probability will be set to a value in (p_low, p_high)
    o:  int
        target qubit. An error of type error_type will be made more likely.

    n_qubits: int
        number of qubits
    
    p_low: float, optional
    Returns

    ---------------
    returns channel_probs except that the probability of an error of the 
        given error type on qubit o has probability p_high
        
    channel_probs must be defined with respect to an error in three-block notation, where each individual qubit
    error is given by e=(e_z|e_x|e_y)"""
    
    channel_probs = np.asarray(channel_probs, dtype=float).copy()
    rng = np.random.default_rng()
    if error_type=='Z':
        channel_probs[o] = rng.uniform(p_low, p_high)  # Z error 
        channel_probs[o+2* n_qubits]=0
        channel_probs[o+n_qubits]= 0

    elif error_type=='X':
        channel_probs[o]=0
        channel_probs[o+2* n_qubits]=0
        channel_probs[o+n_qubits]= rng.uniform(p_low, p_high)

    elif error_type=='Y':
        channel_probs[o]=0
        channel_probs[o+n_qubits]=0
        channel_probs[o +2* n_qubits] =rng.uniform(p_low, p_high)

    
    return channel_probs



def find_first_short_logical_in_coset(T, logi_commute, logi_anticommute,
                                       remaining_logicals:list=None, o=None,
                                       max_iter=1000, channel_probs=None,iter_per_e:int=10):
     
    """Finds a short logical of a tableau T that is in the same logical coset as logi_commute and anti-
    commutes with logi_anticommute. Option to add remaining logicals as argument, acting on the other logical qubits 
    than logi_commute and logi_anticommute. Outputs a first_short_logical that COMMUTES with all remaining logicals. 
    Remaining logicals must be a SYMPLECTIC BASIS"""
 
    if sc.rank_F2(T) < T.shape[0]:
        raise ValueError('T not full rank')

    n_qubits = T.shape[1] // 2
    T = ta.to_gf2_tableau(T)

    logi_anticommute = ta.to_gf2_tableau(logi_anticommute)
    logi_commute = ta.to_gf2_tableau(logi_commute)
    remaining_logicals = [ta.to_gf2_tableau(r) for r in remaining_logicals]

    T_aug = T
    
    for r in remaining_logicals:
         
        T_aug = sc.append_logical_to_tableau(T_aug, r)

    T_aug = sc.append_logical_to_tableau(T_aug, logi_commute)
    T_aug = sc.append_logical_to_tableau(T_aug, logi_anticommute)
    T_aug = ta.turn_tableau_into_TXZY(T_aug)

    r = sc.rank_F2(T_aug)
    if r < T_aug.shape[0]:
        print(f"T_aug rank deficient after appends: {r} < {T_aug.shape[0]}")

    syndrome = np.zeros(T_aug.shape[0], dtype=int)
    syndrome[-1] = 1
    syndrome_1d = syndrome.astype(np.uint8)
         
    if channel_probs is None:
        channel_probs = np.full(3 * n_qubits, 0.1)

    
    if o is not None:
        rng = np.random.default_rng()

        for e in ['Z','X','Y']:

            for i in range(iter_per_e):

                cp = rng.uniform(0.05, 0.45, size=3 * n_qubits)
                cp = bias_channel_toward_qubit(cp, o, e, n_qubits)
                
                #obtain channel_probs that
                #bias error towards error of type e on qubit o

                decoder = BpOsdDecoder(T_aug, channel_probs=cp,
                            max_iter=max_iter, bp_method="ms",
                            osd_method="osd_cs", osd_order=6)
                result = decoder.decode(syndrome_1d)

                if result is not None:
                     
                    result_tab = ta.turn_TXZYerror_into_tableau(result.reshape(1, -1)) #turn the result back to symplectic form
                    v = np.asarray(result_tab).ravel().astype(int)

                    if (v[o] or v[o + n_qubits]): #check whether result has support on qubit o
                        result=result
                        break

                    else:
                        #nonzero = np.flatnonzero(np.asarray(result))
                        #channel_probs[nonzero] = 1e-4   # punish exactly the columns the decoder used
                        result=None

            if result is not None:
                break
 

        if result is None or not np.asarray(result).any():
            return None
    else:
        decoder = BpOsdDecoder(T_aug, channel_probs=channel_probs,
                            max_iter=max_iter, bp_method="ms",
                            osd_method="osd_cs", osd_order=6)
        
        result = decoder.decode(syndrome_1d)

    if result is None:
        return None
    else:
        actual = np.asarray(T_aug @ GF2(result.reshape(-1, 1))).ravel()
        if not np.array_equal(np.asarray(actual).astype(int), np.asarray(syndrome_1d).astype(int)):
            return None #check whether decoder returned correct syndrome

        result_tab = ta.turn_TXZYerror_into_tableau(result.reshape(1, -1))
 
        v = np.asarray(result_tab).ravel().astype(int)

        return ta.to_gf2_tableau(v.reshape(1, -1))
 


def find_short_second_logical_in_coset(T,logi_anti_commute,logi_commute,additional_logicals:list=[],\
                              max_iter:int=100):

    #finds second logical with hopefully small overlap
    #additional_logicals must be a symplectic basis!

    if sc.rank_F2(T)<T.shape[0]:

        raise ValueError('T not full rank')

    n_qubits=T.shape[1]//2 #integer division

    T=ta.to_gf2_tableau(T)

    if isinstance(logi_anti_commute,list):
        logi_anti_commute=sc.tableau_list_to_matrix(logi_anti_commute)

    if isinstance(logi_commute,list):
        logi_commute=sc.tableau_list_to_matrix(logi_commute)

    
    for a in additional_logicals:
         
        T=sc.append_logical_to_tableau(T,ta.to_gf2_tableau(a))

    T=sc.append_logical_to_tableau(T,logi_commute)
    T=sc.append_logical_to_tableau(T,logi_anti_commute)
 
    T=ta.turn_tableau_into_TXZY(T)

    syndrome = np.zeros(T.shape[0], dtype=int)
    syndrome[-1]=1 #assumes logical qubit is last row
    syndrome_1d = syndrome.flatten()

    syndrome_1d = syndrome_1d.astype(np.uint8)

 
    channel_probs =create_error_probs_from_logical(n_qubits,logi_anti_commute,p_base=0.1,p_punish=0.01)

    decoder = BpOsdDecoder(T, channel_probs=channel_probs,
                                max_iter=max_iter, bp_method="ms",
                                osd_method="osd_cs", osd_order=6)

    result = decoder.decode(syndrome_1d)

    if result is None or not np.asarray(result).any():
        return None

    else:

        assert np.array_equal((T @ GF2(result)), syndrome_1d), "decoded result does not reproduce the syndrome"
        assert result.any(), "decoder returned all-zero (no operator found)"

    result=result.reshape(1, -1)

    return ta.turn_TXZYerror_into_tableau(result)



def main():

    numq = 400
    g = nx.erdos_renyi_graph(numq, 0.7)
    #g = nx.cycle_graph(numq)
    g = nx.to_numpy_array(g, dtype = np.uint16)

    xlogi, zlogi, stabi = create_graph_code(g)
    print('xlogical input: ',tableau2paulistring(xlogi))
    #for i in range(stabi.shape[0]):
        #print(tableau2paulistring(stabi[i,:]))

    T=sc.tableau_list_to_matrix(stabi)
    #X_logicals,Z_logicals,logicals=find_logical_op_basis(T,n_qubits)
    lost_qubits=[0,1]
    short_z=find_short_first_logical(T,xlogi)
    #x=find_short_second_logical(T,short_z)
    nonzero=np.flatnonzero(short_z.ravel())
    #print('nonzero indices', nonzero)
    print('short zlogical from decoder: ',tableau2paulistring(short_z))
    #print('xlogical decoder',tableau2paulistring(x))
    short_x=sc.find_anti_commuting_logi_at_O(T,short_z,nonzero[0])
    print('xlogical anticommuting at {}, linear algebra method: '.format(nonzero[0]),(tableau2paulistring(short_x)))

    return short_z




if __name__ == "__main__":

    main()



